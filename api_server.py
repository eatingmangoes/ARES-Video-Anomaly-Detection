import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel  
import redis
import json
import uuid
import base64
import logging
import os


REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = 6379
QUEUE_NAME = "video_queue"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - API - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"FATAL: Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Is it running? Error: {e}")
    
    raise

app = FastAPI()

@app.post("/classify_video/")
async def enqueue_classification(video: UploadFile = File(...)):
    """Accepts a video, adds it to the processing queue, and returns a job ID instantly."""
    job_id = f"job:{uuid.uuid4()}"
    
    video_content = await video.read()
    video_base64 = base64.b64encode(video_content).decode('utf-8')

    job_data = {
        "job_id": job_id,
        "video_data_base64": video_base64
    }

    try:
        redis_client.lpush(QUEUE_NAME, json.dumps(job_data))
        logger.info(f"Queued job {job_id}")
        return {"job_id": job_id, "status": "queued"}
    except redis.exceptions.RedisError as e:
        logger.error(f"Failed to queue job: {e}")
        raise HTTPException(status_code=500, detail="Could not queue the job.")

@app.get("/get_result/{job_id}")
async def get_result(job_id: str):
    """Checks for the result of a completed job using the job ID."""
    try:
        result_data = redis_client.get(job_id)
        if result_data:
            logger.info(f"Result found for job {job_id}")
            return {"job_id": job_id, "status": "completed", "result": json.loads(result_data)}
        else:
            return {"job_id": job_id, "status": "pending"}
    except redis.exceptions.RedisError as e:
        logger.error(f"Failed to retrieve result: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve the result.")


class Feedback(BaseModel):
    request_id: str
    correct_class: str
    is_new_class: bool = False

@app.post("/feedback/")
async def receive_feedback(feedback: Feedback):
    
    
    
    EXPERIENCE_LOG_FILE = "logs/rl_experiences.jsonl"
    updated_logs, found = [], False
    try:
        with open(EXPERIENCE_LOG_FILE, "r") as f:
            for line in f:
                log_entry = json.loads(line)
                if log_entry["request_id"] == feedback.request_id:
                    found = True
                    log_entry["reward"] = -1.0
                    log_entry["correct_action"] = f"CLASS: {feedback.correct_class}"
                updated_logs.append(log_entry)
        if not found: raise HTTPException(status_code=404, detail="Request ID not found.")
        with open(EXPERIENCE_LOG_FILE, "w") as f:
            for entry in updated_logs: f.write(json.dumps(entry) + "\n")
    except FileNotFoundError:
        
        with open(EXPERIENCE_LOG_FILE, "w") as f:
            f.write("") 
        return {"status": "feedback received, log created", "request_id": feedback.request_id}
    return {"status": "feedback received", "request_id": feedback.request_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8289)