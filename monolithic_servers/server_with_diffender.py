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
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionInpaintPipeline
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
from peft import PeftModel

# Import the new utility
from diffender_utils import run_diffender

# --- Configuration ---
# Models
CAPTION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
LLM_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
FINE_TUNED_ADAPTER_PATH = "llama2-13b-anomaly-adapter"

# Performance & Hardware
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
VIDEO_SAMPLE_RATE_HZ = 1

# DIFFender Settings
APPLY_DIFFENDER = True # Set to False to disable the defense layer
DIFFENDER_GENERIC_PROMPT = "a high-quality photograph of a scene" # Generic prompt for defense

# Logging & Data (Unchanged)
LOG_DIR = "logs"
EXPERIENCE_LOG_FILE = os.path.join(LOG_DIR, "rl_experiences.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "server_rl_video.log")), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Anomaly Class Management (Unchanged) ---
ANOMALY_CLASSES: List[str] = [
    "Abuse", "Arrest", "Arson", "Assault", "Brawling", "Burglary", "Chasing",
    "Explosion", "Fighting", "Normal", "Road Accidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Snatching", "Vandalism", "Violence",
    "Other Abnormalities"
]
LEARNED_CLASSES: Set[str] = set()

def get_current_classes() -> List[str]:
    return sorted(list(set(ANOMALY_CLASSES) | LEARNED_CLASSES))
def get_classes_for_prompt() -> str:
    return ", ".join([f"'{cls}'" for cls in get_current_classes()])

# --- Prompts (Unchanged) ---
SYSTEM_PROMPT_CLASSIFY_RL = """You are an AI assistant specialized in analyzing security footage descriptions.
Your task is to classify the described scene into ONE of the following categories: {classes_for_prompt}.
Your response MUST start with the keyword "CLASS:" followed by a single space and then ONLY the chosen category name from the list.
You receive feedback on your classifications and must adapt your future responses. Some objects, while typically normal, may be considered anomalies in specific contexts based on that feedback."""
USER_INSTRUCTION_TEMPLATE_CLASSIFY = """Scene Description: {aggregated_captions}

Choose ONE category from the provided list that best describes this scene and format your response as "CLASS: [Chosen Category]"."""

# --- Lifespan for Model Loading/Unloading ---
ml_models: Dict[str, Any] = {}

def load_diffender_models(device, dtype):
    """Loads all necessary models for DIFFender."""
    logger.info("Loading DIFFender artifacts...")
    models = {}
    
    # --- CHANGE 1: Use v1.5 components for consistency with the inpainting model ---
    base_model_path = "runwayml/stable-diffusion-v1-5"
    inpainting_model_path = "runwayml/stable-diffusion-inpainting"
    
    models['diffender_vae'] = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype).to(device)
    models['diffender_unet'] = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", torch_dtype=dtype).to(device)
    
    # These CLIP models are standard and don't need to be changed
    models['diffender_tokenizer'] = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    models['diffender_text_encoder'] = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to(device)
    
    models['diffender_scheduler'] = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    
    # --- CHANGE 2: Remove the incorrect revision='fp16' argument ---
    models['diffender_pipe'] = StableDiffusionInpaintPipeline.from_pretrained(
        inpainting_model_path,
        torch_dtype=dtype
    ).to(device)
    
    logger.info("DIFFender artifacts loaded successfully.")
    return models

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Starting model loading (RL fine-tuned video mode)...")
    load_start_time = time.time()
    global ml_models
    try:
        # 1. Load Captioning Model
        logger.info(f"Loading caption processor: {CAPTION_MODEL_NAME}")
        ml_models["caption_processor"] = Blip2Processor.from_pretrained(CAPTION_MODEL_NAME)
        logger.info(f"Loading caption model: {CAPTION_MODEL_NAME}")
        ml_models["caption_model"] = Blip2ForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_NAME, torch_dtype=DTYPE
        ).to(DEVICE)
        ml_models["caption_model"].eval()

        # 2. Load LLM
        logger.info(f"Loading base LLM: {LLM_MODEL_NAME}")
        base_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=DTYPE, device_map="auto")
        if os.path.exists(FINE_TUNED_ADAPTER_PATH):
            logger.info(f"Applying adapter from: {FINE_TUNED_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(base_model, FINE_TUNED_ADAPTER_PATH)
            tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_ADAPTER_PATH, use_fast=False)
        else:
            logger.warning(f"No fine-tuned adapter found. Using base model: {LLM_MODEL_NAME}")
            model = base_model
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=False)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        ml_models["llm_model"] = model.eval()
        ml_models["llm_tokenizer"] = tokenizer

        # 3. *** NEW: Load DIFFender Models ***
        if APPLY_DIFFENDER:
            diffender_models = load_diffender_models(DEVICE, DTYPE)
            ml_models.update(diffender_models)

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

# --- Helper Function to Parse Class (Unchanged) ---
def parse_class_label(llm_output: str, valid_classes: List[str]) -> str:
    llm_output_cleaned = llm_output.strip()
    match = re.search(r"CLASS:\s*(.+)", llm_output_cleaned, re.IGNORECASE)
    if match:
        potential_class = match.group(1).strip().replace("'", "").replace('"', "")
        for valid_class in valid_classes:
            if potential_class.lower() == valid_class.lower():
                return valid_class
        logger.warning(f"LLM output '{potential_class}' not in valid classes.")
        return "Unknown"
    else:
        for valid_class in valid_classes:
            if re.search(r'\b' + re.escape(valid_class) + r'\b', llm_output_cleaned, re.IGNORECASE):
                return valid_class
        return "Unknown"

@app.post("/classify_video/")
async def classify_video_endpoint(video: UploadFile = File(...)):
    if not ml_models.get("loaded", False):
        raise HTTPException(status_code=503, detail="Models are not available or failed to load.")

    request_id = str(uuid.uuid4())
    processing_times = {}
    total_start_time = time.time()

    # --- Step 1: Video Processing (Extract Frames) - Unchanged ---
    video_proc_start = time.time()
    pil_images = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        contents = await video.read()
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened(): raise HTTPException(status_code=400, detail="Could not open video file.")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, round(video_fps / VIDEO_SAMPLE_RATE_HZ))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb_frame))
            frame_count += 1
        cap.release()
        logger.info(f"Extracted {len(pil_images)} frames from video.")
    except Exception as e:
        logger.error(f"Error processing video: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {e}")
    finally:
        os.unlink(tmp_file_path)

    if not pil_images:
        raise HTTPException(status_code=400, detail="No frames could be extracted from the video.")
    processing_times["video_extraction_seconds"] = round(time.time() - video_proc_start, 2)

    # --- Step 2: *** NEW: Defend Frames with DIFFender *** ---
    defended_images = []
    if APPLY_DIFFENDER:
        defense_start = time.time()
        logger.info(f"Applying DIFFender to {len(pil_images)} frames...")
        try:
            for i, frame in enumerate(pil_images):
                logger.info(f"Defending frame {i+1}/{len(pil_images)}...")
                defended_frame = run_diffender(
                    init_img=frame,
                    rp=DIFFENDER_GENERIC_PROMPT,
                    qp=DIFFENDER_GENERIC_PROMPT,
                    models=ml_models
                )
                defended_images.append(defended_frame)
            logger.info("DIFFender processing complete.")
            processing_times["diffender_defense_seconds"] = round(time.time() - defense_start, 2)
        except Exception as e:
            logger.error(f"Error during DIFFender defense: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Frame defense failed.")
    else:
        logger.info("Skipping DIFFender defense.")
        defended_images = pil_images # Use original images if defense is off

    # --- Step 3: Image Captioning (BLIP-2) ---
    captioning_start = time.time()
    captions = []
    try:
        with torch.no_grad():
            for i in range(0, len(defended_images), BATCH_SIZE):
                batch = defended_images[i:i+BATCH_SIZE]
                inputs = ml_models["caption_processor"](
                    images=batch, padding=True, return_tensors="pt"
                ).to(DEVICE, DTYPE)
                generated_ids = ml_models["caption_model"].generate(**inputs, max_new_tokens=50)
                generated_texts = ml_models["caption_processor"].batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                captions.extend([text.strip() for text in generated_texts])
    except Exception as e:
        logger.error(f"Error during captioning: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Captioning failed.")
    processing_times["captioning_seconds"] = round(time.time() - captioning_start, 2)

    # --- Step 4: LLM Classification (Llama 2) - Unchanged ---
    llm_start = time.time()
    aggregated_captions = ". ".join(list(set(captions)))
    current_classes = get_current_classes()
    classes_prompt_str = get_classes_for_prompt()
    system_prompt = SYSTEM_PROMPT_CLASSIFY_RL.format(classes_for_prompt=classes_prompt_str)
    user_instruction = USER_INSTRUCTION_TEMPLATE_CLASSIFY.format(aggregated_captions=aggregated_captions)
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_instruction} [/INST]"
    
    try:
        inputs = ml_models["llm_tokenizer"](prompt, return_tensors="pt").to(ml_models["llm_model"].device)
        with torch.no_grad():
            output = ml_models["llm_model"].generate(
                **inputs, max_new_tokens=20, temperature=0.01, do_sample=False
            )
        prompt_len = inputs.input_ids.shape[-1]
        llm_output_only = ml_models["llm_tokenizer"].decode(output[0][prompt_len:], skip_special_tokens=True)
        predicted_class = parse_class_label(llm_output_only, current_classes)
    except Exception as e:
        logger.error(f"Error during LLM classification: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="LLM classification failed.")
    processing_times["llm_classification_seconds"] = round(time.time() - llm_start, 2)

    # --- Log experience for RL (Unchanged) ---
    experience = {
        "request_id": request_id,
        "state": prompt,
        "action": llm_output_only,
        "predicted_class": predicted_class,
        "reward": None
    }
    with open(EXPERIENCE_LOG_FILE, "a") as f:
        f.write(json.dumps(experience) + "\n")
        
    processing_times["total_seconds"] = round(time.time() - total_start_time, 2)
    logger.info(f"Request {request_id} processed in {processing_times['total_seconds']}s. Class: {predicted_class}")
    
    return {
        "request_id": request_id,
        "predicted_class": predicted_class,
        "processing_times": processing_times
    }

# --- Feedback Endpoint and Main Execution (Unchanged) ---
class Feedback(BaseModel):
    request_id: str
    correct_class: str
    is_new_class: bool = False

@app.post("/feedback/")
async def receive_feedback(feedback: Feedback):
    if feedback.is_new_class:
        LEARNED_CLASSES.add(feedback.correct_class)
        logger.info(f"New class added via feedback: '{feedback.correct_class}'. Current learned: {LEARNED_CLASSES}")
    updated_logs = []
    found = False
    try:
        with open(EXPERIENCE_LOG_FILE, "r") as f:
            for line in f:
                log_entry = json.loads(line)
                if log_entry["request_id"] == feedback.request_id:
                    found = True
                    if log_entry["predicted_class"].lower() == feedback.correct_class.lower():
                        log_entry["reward"] = 1.0
                    else:
                        log_entry["reward"] = -1.0
                    log_entry["correct_action"] = f"CLASS: {feedback.correct_class}"
                updated_logs.append(log_entry)
        if not found:
            raise HTTPException(status_code=404, detail="Request ID not found in experience log.")
        with open(EXPERIENCE_LOG_FILE, "w") as f:
            for entry in updated_logs:
                f.write(json.dumps(entry) + "\n")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Experience log file not found.")
    return {"status": "feedback received", "request_id": feedback.request_id}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5900))
    uvicorn.run("server_with_diffender:app", host="0.0.0.0", port=port, reload=True)