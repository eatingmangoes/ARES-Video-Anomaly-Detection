import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import redis
import json
import base64
import time
import os
import re
import cv2
from PIL import Image, ImageFilter
import io
import tempfile
import logging
import uuid
import traceback
from typing import List
import numpy as np

# --- Configuration ---
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = 6379
QUEUE_NAME = "video_queue"

# Use Llama-2 13B as established
LLM_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
FINE_TUNED_ADAPTER_PATH = "logs/llama2-13b-anomaly-adapter"
CAPTION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"

DTYPE = torch.float16
CAPTION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
VIDEO_SAMPLE_RATE_HZ = 1

LOG_DIR = "logs"
EXPERIENCE_LOG_FILE = os.path.join(LOG_DIR, "rl_experiences.jsonl")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Anomaly Class Definitions ---
ANOMALY_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
    "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo"
]
LEARNED_CLASSES = set()

def get_current_classes() -> List[str]:
    return sorted(list(set(ANOMALY_CLASSES) | LEARNED_CLASSES))

def get_classes_for_prompt() -> str:
    return ", ".join([f"'{cls}'" for cls in get_current_classes()])

# --- Prompts ---
SYSTEM_PROMPT_CLASSIFY_RL = """You are an intelligent video classification assistant.
Your task is to analyze the described scene and classify it into exactly ONE of the following categories: {classes_for_prompt}.
Your response MUST start with the keyword "CLASS:" followed by a single space and then ONLY the chosen category name.
Base your decision on the visual actions described."""

USER_INSTRUCTION_TEMPLATE_CLASSIFY = """Scene Description: {aggregated_captions}

Choose ONE category from the provided list that best describes this scene and format your response as "CLASS: [Chosen Category]"."""

def parse_class_label(llm_output: str, valid_classes: List[str]) -> str:
    llm_output_cleaned = llm_output.strip()
    match = re.search(r"CLASS:\s*(.+)", llm_output_cleaned, re.IGNORECASE)
    if match:
        potential_class = match.group(1).strip().replace("'", "").replace('"', "")
        for valid_class in valid_classes:
            if potential_class.lower() == valid_class.lower():
                return valid_class
    for valid_class in valid_classes:
        if re.search(r'\b' + re.escape(valid_class) + r'\b', llm_output_cleaned, re.IGNORECASE):
            return valid_class
    return "Unknown"

# --- NEW: Adversarial Defense Module ---
class AdversarialShield:
    """
    Implements Input Transformation defenses to purify images before 
    they are processed by the AI models.
    """
    def __init__(self):
        logger.info("Adversarial Shield initialized.")

    def purify_batch(self, pil_images: List[Image.Image]) -> List[Image.Image]:
        """
        Applies a series of non-differentiable transformations to break 
        adversarial noise patterns.
        """
        purified_images = []
        for img in pil_images:
            # 1. JPEG Compression Defense
            # Adversarial noise is often high-frequency and fragile. 
            # Saving as JPEG discards this noise.
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75) # Quality 75 is a sweet spot
            buffer.seek(0)
            img_jpeg = Image.open(buffer)

            # 2. Spatial Smoothing (Median Filter)
            # Effective against "Patch Attacks" (like in your PDF) and salt-and-pepper noise.
            # It replaces a pixel with the median of its neighbors, eroding small artificial patches.
            img_smooth = img_jpeg.filter(ImageFilter.MedianFilter(size=3))

            # 3. Bit-Depth Reduction (Posterization) - Optional but effective
            # Reduces the color palette, snapping precise adversarial gradients to the nearest safe color.
            # img_final = img_smooth.quantize(colors=128).convert('RGB')
            
            purified_images.append(img_smooth)
        
        return purified_images

# --- Main Worker Functions ---

def load_models():
    logger.info("Worker starting model loading...")
    ml_models = {}
    
    logger.info(f"Loading captioner: {CAPTION_MODEL_NAME}")
    # Use use_fast=False to avoid tokenizer errors
    ml_models["caption_processor"] = Blip2Processor.from_pretrained(CAPTION_MODEL_NAME, use_fast=False)
    ml_models["caption_model"] = Blip2ForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_NAME, torch_dtype=DTYPE
    ).to(CAPTION_DEVICE)

    logger.info(f"Loading LLM: {LLM_MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    if os.path.exists(FINE_TUNED_ADAPTER_PATH):
        logger.info(f"Applying adapter from: {FINE_TUNED_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, FINE_TUNED_ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_ADAPTER_PATH)
    else:
        logger.warning("No adapter found. Using base model.")
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    ml_models["llm_model"] = model
    ml_models["llm_tokenizer"] = tokenizer
    
    # Initialize the Defense Shield
    ml_models["defense_shield"] = AdversarialShield()
    
    logger.info("Worker models loaded successfully.")
    return ml_models

def process_video_job(video_bytes: bytes, ml_models: dict):
    request_id_for_feedback = str(uuid.uuid4())
    pil_images, captions = [], []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, round(fps / VIDEO_SAMPLE_RATE_HZ)) if fps and fps > 0 else 1
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % interval == 0:
                pil_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            count += 1
        cap.release()
    finally:
        os.unlink(tmp_file_path)

    if not pil_images:
        return {"error": "No frames could be extracted from the video."}

    # --- DEFENSE STEP: Purify Frames ---
    # Before the AI sees the images, we clean them.
    # This removes adversarial noise and patches.
    logger.info("Applying Adversarial Defense Shield...")
    purified_images = ml_models["defense_shield"].purify_batch(pil_images)

    # Generate captions using the purified images
    with torch.no_grad():
        for i in range(0, len(purified_images), BATCH_SIZE):
            inputs = ml_models["caption_processor"](
                images=purified_images[i:i+BATCH_SIZE], padding=True, return_tensors="pt"
            ).to(CAPTION_DEVICE, DTYPE)
            ids = ml_models["caption_model"].generate(**inputs, max_new_tokens=50)
            captions.extend([text.strip() for text in ml_models["caption_processor"].batch_decode(ids, skip_special_tokens=True)])
    
    # Classify with LLM (Standard Llama-2 Prompt)
    aggregated_captions = ". ".join(list(set(captions)))
    system_prompt = SYSTEM_PROMPT_CLASSIFY_RL.format(classes_for_prompt=get_classes_for_prompt())
    user_instruction = USER_INSTRUCTION_TEMPLATE_CLASSIFY.format(aggregated_captions=aggregated_captions)
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_instruction} [/INST]"
    
    inputs = ml_models["llm_tokenizer"](prompt, return_tensors="pt").to(ml_models["llm_model"].device)
    with torch.no_grad():
        output = ml_models["llm_model"].generate(**inputs, max_new_tokens=20, temperature=0.01)
    
    predicted_class = parse_class_label(
        ml_models["llm_tokenizer"].decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True),
        get_current_classes()
    )

    experience = {
        "request_id": request_id_for_feedback,
        "state": prompt,
        "action": f"CLASS: {predicted_class}",
        "predicted_class": predicted_class,
        "reward": None
    }
    try:
        with open(EXPERIENCE_LOG_FILE, "a") as f:
            f.write(json.dumps(experience) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to experience log: {e}")
        
    return {
        "predicted_class": predicted_class,
        "request_id_for_feedback": request_id_for_feedback
    }

if __name__ == "__main__":
    try:
        ml_models = load_models()
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        redis_client.ping()
    except Exception as e:
        logger.error(f"FATAL: Worker could not initialize. Shutting down. Error: {e}")
        logger.error(traceback.format_exc())
        exit()

    logger.info(f"Worker started successfully. Waiting for jobs on queue '{QUEUE_NAME}'...")
    while True:
        try:
            job_data_bytes = redis_client.brpop(QUEUE_NAME, timeout=0)[1]
            job_data = json.loads(job_data_bytes)
            job_id = job_data["job_id"]
            logger.info(f"Processing job {job_id}...")
            start_time = time.time()
            
            video_bytes = base64.b64decode(job_data["video_data_base64"])
            result = process_video_job(video_bytes, ml_models)
            
            processing_time = time.time() - start_time
            result["processing_time_seconds"] = round(processing_time, 2)
            redis_client.set(job_id, json.dumps(result), ex=3600)
            logger.info(f"Job {job_id} finished in {processing_time:.2f}s. Result: {result.get('predicted_class', 'ERROR')}")

        except Exception as e:
            logger.error(f"CRITICAL: Error processing a job. Skipping. Error: {e}", exc_info=True)
            time.sleep(1)