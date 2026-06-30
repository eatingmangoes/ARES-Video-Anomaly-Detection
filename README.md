# ARES: Adaptive Real-time Event Surveillance

[![Paper](https://img.shields.io/badge/Paper-IEEE_AVSS_2026_(Under_Review)-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)]()

This is the official code repository for the paper: **"ARES: A Memory-Augmented Visual-Semantic Framework for Adaptive Real-time Video Anomaly Detection via Distributed Foundation Models."**

This repository provides a distributed pipeline for real-time video anomaly detection using Vision-Language Models (BLIP-2) and Large Language Models (Llama-2). It is designed to handle high-latency LLM inference without blocking video ingestion, and includes a Human-in-the-Loop (HITL) fine-tuning script to adapt the model to custom security rules.

> **Note on the Visual Memory Module:** 
> This initial release contains the distributed inference framework, the semantic reasoning pipeline, and the PEFT/LoRA fine-tuning scripts. We are currently refactoring the FAISS-based visual memory module ($S_{vis}$) to remove local dependencies and will push it to this repository soon. 

## Repository Structure

* `api_server.py`: FastAPI server that receives video files and pushes jobs to a Redis queue.
* `ai_worker.py`: Background worker that pulls jobs, extracts frames, generates captions (BLIP-2), and classifies them (Llama-2).
* `ai_worker_adversarial.py`: Same as the standard worker, but includes a lightweight preprocessing step to defend against adversarial patch attacks.
* `train_sft_model.py`: Script to fine-tune the LLM using LoRA (4-bit) based on user feedback.
* `benchmark_ucf.py`: Script used to test against the UCF-Crime dataset and simulate PGD attacks.
* `generate_plots.py`: Generates the matplotlib graphs used in the paper.
* `monolithic_servers/`: Contains older, synchronous versions of the code used for latency baseline comparisons.

## Setup Instructions

**1. Environment Setup**  
You need a CUDA-capable GPU (at least 24GB VRAM for the 13B model).
```bash
git clone https://github.com/eatingmangoes/ARES-Video-Anomaly-Detection.git
cd ARES-Video-Anomaly-Detection
pip install -r requirements.txt
```

**2. Start Redis**  
The system relies on Redis for the job queue.
* Linux: `sudo apt install redis-server && sudo systemctl start redis-server`
* Docker: `docker run -p 6379:6379 -d redis`

**3. HuggingFace Access**  
Because Llama-2 is a gated model, authenticate your machine using the HuggingFace CLI:
```bash
huggingface-cli login
```

## Running the System

You will need to run the API and the worker in separate terminals.

**Terminal 1: Start the API**
```bash
python api_server.py
```

**Terminal 2: Start a Worker**
```bash
# Standard worker
python ai_worker.py

# Or the worker with adversarial defense
python ai_worker_adversarial.py
```

**Terminal 3: Send a Request**
```bash
curl -X POST -F "video=@your_video_file.mp4" http://localhost:8289/classify_video/
```
The API will return a `job_id` immediately. The worker will pick it up and log the classification results in its terminal.

## Human-in-the-Loop Fine-Tuning

If you want the model to learn a specific rule for your environment (e.g., "classify a red backpack as an anomaly here"), you can log corrections and train a LoRA adapter.

**1. Log Feedback:**
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"request_id": "THE_JOB_ID", "correct_class": "Anomaly", "is_new_class": true}' \
http://localhost:8289/feedback/
```
This saves the corrected state/action to `logs/rl_experiences.jsonl`.

**2. Train the Adapter:**
Once you have logged a few corrections, run the training script:
```bash
python train_sft_model.py
```
This will train a low-rank adapter (PEFT) and save it to `llama2-13b-anomaly-adapter/`. The next time you start `ai_worker.py`, it will automatically detect and load these updated weights.

## Reproducing Paper Results

To generate the plots used in the paper (scalability, ablation, etc.):
```bash
python generate_plots.py
```
The images will be saved in the `paper_plots/` directory.

To run the adversarial benchmark script:
```bash
python benchmark_ucf.py
```
