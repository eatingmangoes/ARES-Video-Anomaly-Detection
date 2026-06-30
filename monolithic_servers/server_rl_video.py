import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, AsyncGenerator, Dict, Any, Set
from contextlib import asynccontextmanager
import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import io
from PIL import Image
import re
import logging
import time
import traceback
import os
import uuid
import json
import cv2
import numpy as np
import tempfile



LLM_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
FINE_TUNED_ADAPTER_PATH = "llama2-13b-anomaly-adapter"


CAPTION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
DTYPE = torch.float16
CAPTION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
VIDEO_SAMPLE_RATE_HZ = 1


LOG_DIR = "logs"
EXPERIENCE_LOG_FILE = os.path.join(LOG_DIR, "rl_experiences.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "server_rl_video.log")), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


ANOMALY_CLASSES: List[str] = ["Abuse", "Arrest", "Arson", "Assault", "Brawling", "Burglary", "Chasing", "Explosion", "Fighting", "Normal", "Other Abnormalities", "Road Accidents", "Robbery", "Shooting", "Shoplifting", "Snatching", "Stealing", "Vandalism", "Violence"]
LEARNED_CLASSES: Set[str] = set()

def get_current_classes() -> List[str]:
    return sorted(list(set(ANOMALY_CLASSES) | LEARNED_CLASSES))

def get_classes_for_prompt() -> str:
    return ", ".join([f"'{cls}'" for cls in get_current_classes()])


SYSTEM_PROMPT_CLASSIFY_RL = """You are an AI assistant specialized in analyzing security footage descriptions.
Your task is to classify the described scene into ONE of the following categories: {classes_for_prompt}.
Your response MUST start with the keyword "CLASS:" followed by a single space and then ONLY the chosen category name from the list.
You receive feedback on your classifications and must adapt your future responses. Some objects, while typically normal, may be considered anomalies in specific contexts based on that feedback."""

USER_INSTRUCTION_TEMPLATE_CLASSIFY = """Scene Description: {aggregated_captions}

Choose ONE category from the provided list that best describes this scene and format your response as "CLASS: [Chosen Category]"."""


ml_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Starting model loading (13B Mode)...")
    load_start_time = time.time()
    global ml_models
    try:
        
        logger.info(f"Loading caption processor: {CAPTION_MODEL_NAME}")
        
        
        ml_models["caption_processor"] = Blip2Processor.from_pretrained(CAPTION_MODEL_NAME, use_fast=False)
        
        logger.info(f"Loading caption model: {CAPTION_MODEL_NAME}")
        ml_models["caption_model"] = Blip2ForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_NAME, torch_dtype=DTYPE
        ).to(CAPTION_DEVICE)
        ml_models["caption_model"].eval()

        
        logger.info(f"Loading base LLM: {LLM_MODEL_NAME} with 4-bit quantization")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )

        
        if os.path.exists(FINE_TUNED_ADAPTER_PATH):
            logger.info(f"Found fine-tuned adapter. Applying adapter from: {FINE_TUNED_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(base_model, FINE_TUNED_ADAPTER_PATH)
            tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_ADAPTER_PATH, use_fast=False)
            logger.info("Successfully applied PEFT adapter.")
        else:
            logger.warning(f"No fine-tuned adapter found. Using base model: {LLM_MODEL_NAME}")
            model = base_model
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=False)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        ml_models["llm_model"] = model
        ml_models["llm_tokenizer"] = tokenizer
        ml_models["llm_model"].eval()
        logger.info(f"LLM is ready for inference.")
        
        ml_models["loaded"] = True
        logger.info(f"All models loaded in {time.time() - load_start_time:.2f}s.")
    except Exception as e:
        logger.error(f"Fatal error during model loading: {e}\n{traceback.format_exc()}")
        ml_models["loaded"] = False
    
    yield
    
    logger.info("Shutting down...")
    ml_models.clear()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("Cleanup complete.")

app = FastAPI(lifespan=lifespan)



def parse_class_label(llm_output: str, valid_classes: List[str]) -> str:
    llm_output_cleaned = llm_output.strip()
    match = re.search(r"CLASS:\s*(.+)", llm_output_cleaned, re.IGNORECASE)
    if match:
        potential_class = match.group(1).strip().replace("'", "").replace('"', "")
        for valid_class in valid_classes:
            if potential_class.lower() == valid_class.lower(): return valid_class
        return "Unknown"
    else:
        for valid_class in valid_classes:
            if re.search(r'\b' + re.escape(valid_class) + r'\b', llm_output_cleaned, re.IGNORECASE): return valid_class
        return "Unknown"

@app.post("/classify_video/")
async def classify_video_endpoint(video: UploadFile = File(...)):
    if not ml_models.get("loaded", False): raise HTTPException(status_code=503, detail="Models are not available or failed to load.")
    request_id = str(uuid.uuid4())
    processing_times, pil_images, captions = {}, [], []
    total_start_time = time.time()
    video_proc_start = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        contents = await video.read()
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened(): raise HTTPException(status_code=400, detail="Could not open video file.")
        video_fps, frame_count = cap.get(cv2.CAP_PROP_FPS), 0
        frame_interval = max(1, round(video_fps / VIDEO_SAMPLE_RATE_HZ))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0: pil_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frame_count += 1
        cap.release()
    finally: os.unlink(tmp_file_path)
    if not pil_images: raise HTTPException(status_code=400, detail="No frames could be extracted.")
    processing_times["video_extraction_seconds"] = round(time.time() - video_proc_start, 2)
    captioning_start = time.time()
    with torch.no_grad():
        for i in range(0, len(pil_images), BATCH_SIZE):
            batch = pil_images[i:i+BATCH_SIZE]
            inputs = ml_models["caption_processor"](images=batch, padding=True, return_tensors="pt").to(CAPTION_DEVICE, DTYPE)
            generated_ids = ml_models["caption_model"].generate(**inputs, max_new_tokens=50)
            captions.extend([text.strip() for text in ml_models["caption_processor"].batch_decode(generated_ids, skip_special_tokens=True)])
    processing_times["captioning_seconds"] = round(time.time() - captioning_start, 2)
    llm_start = time.time()
    aggregated_captions = ". ".join(list(set(captions)))
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT_CLASSIFY_RL.format(classes_for_prompt=get_classes_for_prompt())}\n<</SYS>>\n\n{USER_INSTRUCTION_TEMPLATE_CLASSIFY.format(aggregated_captions=aggregated_captions)} [/INST]"
    inputs = ml_models["llm_tokenizer"](prompt, return_tensors="pt").to(ml_models["llm_model"].device)
    with torch.no_grad():
        output = ml_models["llm_model"].generate(**inputs, max_new_tokens=20, temperature=0.01, do_sample=False)
    predicted_class = parse_class_label(ml_models["llm_tokenizer"].decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True), get_current_classes())
    processing_times["llm_classification_seconds"] = round(time.time() - llm_start, 2)
    experience = {"request_id": request_id, "state": prompt, "action": f"CLASS: {predicted_class}", "predicted_class": predicted_class, "reward": None}
    with open(EXPERIENCE_LOG_FILE, "a") as f: f.write(json.dumps(experience) + "\n")
    processing_times["total_seconds"] = round(time.time() - total_start_time, 2)
    logger.info(f"Request {request_id} processed in {processing_times['total_seconds']}s. Class: {predicted_class}")
    return {"request_id": request_id, "predicted_class": predicted_class, "processing_times": processing_times}

class Feedback(BaseModel):
    request_id: str
    correct_class: str
    is_new_class: bool = False

@app.post("/feedback/")
async def receive_feedback(feedback: Feedback):
    if feedback.is_new_class: LEARNED_CLASSES.add(feedback.correct_class)
    updated_logs, found = [], False
    try:
        with open(EXPERIENCE_LOG_FILE, "r") as f:
            for line in f:
                log_entry = json.loads(line)
                if log_entry["request_id"] == feedback.request_id:
                    found = True
                    log_entry["reward"] = -1.0 if log_entry["predicted_class"].lower() != feedback.correct_class.lower() else 1.0
                    log_entry["correct_action"] = f"CLASS: {feedback.correct_class}"
                updated_logs.append(log_entry)
        if not found: raise HTTPException(status_code=404, detail="Request ID not found.")
        with open(EXPERIENCE_LOG_FILE, "w") as f:
            for entry in updated_logs: f.write(json.dumps(entry) + "\n")
    except FileNotFoundError: raise HTTPException(status_code=404, detail="Experience log file not found.")
    return {"status": "feedback received", "request_id": feedback.request_id}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5900))
    uvicorn.run(app, host="0.0.0.0", port=port)