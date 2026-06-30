# train_sft_model.py (Updated for 13B Model with Quantization and PEFT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig
import json
import logging
import os
import time

# --- Configuration ---
LLM_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"  # <-- CHANGED TO 13B
ADAPTER_PATH = "llama2-13b-anomaly-adapter"      # <-- CHANGED TO 13B
EXPERIENCE_LOG_FILE = "logs/rl_experiences.jsonl"
PROCESSED_LOG_PATH = "logs/processed_request_ids.txt"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Load Model and Tokenizer with 4-bit Quantization
logger.info(f"Loading base model and tokenizer: {LLM_MODEL_NAME}")
try:
    # Define quantization configuration for memory savings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config, # <-- ENABLED QUANTIZATION
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model. Error: {e}")
    exit()

# 2. Configure PEFT/LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Load and Prepare the Dataset (No changes here)
def load_and_format_dataset(filepath: str):
    processed_ids = set()
    if os.path.exists(PROCESSED_LOG_PATH):
        with open(PROCESSED_LOG_PATH, 'r') as f:
            processed_ids = set(line.strip() for line in f)
    new_experiences = []
    if not os.path.exists(filepath):
        logger.warning(f"Experience log file not found at: {filepath}. Nothing to train on.")
        return None, set()
    with open(filepath, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("reward") == -1.0 and entry["request_id"] not in processed_ids:
                    text_data = f"{entry['state']} {entry['correct_action']}"
                    new_experiences.append({"text": text_data, "request_id": entry["request_id"]})
            except (json.JSONDecodeError, KeyError):
                continue
    if not new_experiences:
        return None, processed_ids
    return Dataset.from_list(new_experiences), processed_ids

dataset, processed_ids = load_and_format_dataset(EXPERIENCE_LOG_FILE)
if dataset is None:
    logger.info("No new negative feedback to train on. Exiting.")
    exit()
logger.info(f"Loaded {len(dataset)} new experiences to train on.")

# 4. Configure TrainingArguments
training_args = TrainingArguments(
    output_dir=os.path.join(LOG_DIR, "sft_training_output_13b"),
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    logging_steps=1,
    save_strategy="epoch",
    report_to="tensorboard",
    optim="paged_adamw_8bit",
)

# 5. Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    dataset_text_field="text",
)

# 6. Run the Training
logger.info("Starting PEFT Fine-Tuning for 13B model...")
start_time = time.time()
trainer.train()
training_time = time.time() - start_time
logger.info(f"SFT training finished in {training_time:.2f} seconds.")

# 7. Save the final adapter
logger.info(f"Saving LoRA adapter to ./{ADAPTER_PATH}")
if not os.path.exists(ADAPTER_PATH):
    os.makedirs(ADAPTER_PATH)
trainer.save_model(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

all_processed_ids = processed_ids.union(set(dataset["request_id"]))
with open(PROCESSED_LOG_PATH, "w") as f:
    for req_id in sorted(list(all_processed_ids)):
        f.write(req_id + '\n')
logger.info(f"Updated processed request IDs log at: {PROCESSED_LOG_PATH}")