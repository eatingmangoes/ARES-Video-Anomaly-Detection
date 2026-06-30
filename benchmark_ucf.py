import os
import random
import time
import requests
import torch
import torch.nn as nn
import torchvision.models.video as models
from torchvision.io import read_video, write_video
import torchvision.transforms as transforms
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
DATASET_ROOT = "/home/shanmukha/Design_Project/UCF-101" 
API_URL = "http://127.0.0.1:8289"
NUM_SAMPLES = 50 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Attack Settings
EPSILON = 8 / 255
ALPHA = 2 / 255
STEPS = 10

# --- PGD ATTACK HELPER ---
print(f"Loading Surrogate Model (R3D_18) for Attack Generation on {DEVICE}...")
# Note: weights might be named differently depending on torchvision version
# Using the most common modern string
surrogate_model = models.r3d_18(weights='KINETICS400_V1').to(DEVICE)
surrogate_model.eval()

def generate_adv_video(video_path, output_path):
    """
    Reads a video, applies PGD attack using the surrogate model, and saves it.
    """
    # 1. Read Video [T, H, W, C]
    # We use pts_unit='sec' to get accurate timing info
    vframes, _, info = read_video(video_path, pts_unit='sec')
    
    # --- FIX 1: Ensure FPS is a standard Python float ---
    fps = float(info['video_fps']) 
    
    # Preprocess: [T, H, W, C] -> [1, C, T, H, W], 0-1 range
    # Resize to 112x112 for the R3D model
    # --- FIX 2: Add antialias=True to silence warnings ---
    resize = transforms.Resize((112, 112), antialias=True)
    
    video_tensor = vframes.permute(3, 0, 1, 2).float() / 255.0
    video_tensor = resize(video_tensor).unsqueeze(0).to(DEVICE)
    
    # 2. Get Target Label (Untargeted Attack)
    with torch.no_grad():
        init_pred = surrogate_model(video_tensor).argmax()
        
    # 3. PGD Loop
    adv_video = video_tensor.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(STEPS):
        adv_video.requires_grad = True
        outputs = surrogate_model(adv_video)
        loss = loss_fn(outputs, init_pred.unsqueeze(0))
        loss.backward()
        
        adv_video = adv_video.detach() + ALPHA * adv_video.grad.sign()
        delta = torch.clamp(adv_video - video_tensor, min=-EPSILON, max=EPSILON)
        adv_video = torch.clamp(video_tensor + delta, min=0, max=1).detach()
        
    # 4. Save Video
    adv_tensor = adv_video.squeeze(0).cpu()
    # Resize back to standard size for VideoPure (e.g., 224x224)
    resize_back = transforms.Resize((224, 224), antialias=True)
    adv_tensor = resize_back(adv_tensor)
    
    adv_tensor = adv_tensor.permute(1, 2, 3, 0) # C, T, H, W -> T, H, W, C
    adv_tensor = (adv_tensor * 255).byte()
    
    # Pass the fixed 'float' fps here
    write_video(output_path, adv_tensor, fps=fps)

# --- API CLIENT HELPER ---
def get_prediction(video_path):
    """Sends video to API and polls for result."""
    try:
        with open(video_path, "rb") as f:
            # The API expects 'video' key for the file
            files = {"video": f}
            res = requests.post(f"{API_URL}/classify_video/", files=files)
            if res.status_code != 200: 
                print(f"API Error {res.status_code}: {res.text}")
                return "Error"
            job_id = res.json()["job_id"]
        
        # Poll
        for _ in range(30): # Wait up to 60s
            time.sleep(2)
            res = requests.get(f"{API_URL}/get_result/{job_id}")
            data = res.json()
            if data["status"] == "completed":
                return data["result"]["predicted_class"]
        return "Timeout"
    except Exception as e:
        print(f"Connection Error: {e}")
        return "Error"

# --- MAIN BENCHMARK LOOP ---
def run_benchmark():
    print("Scanning dataset...")
    # Matches structure: Root/Class/Video.avi
    all_videos = glob.glob(os.path.join(DATASET_ROOT, "*", "*.avi"))
    
    if not all_videos:
        print("No videos found! Please check DATASET_ROOT path.")
        return

    # Select Subset
    selected_videos = random.sample(all_videos, min(NUM_SAMPLES, len(all_videos)))
    print(f"Selected {len(selected_videos)} videos for benchmarking.")

    clean_correct = 0
    adv_correct = 0
    total = 0

    print(f"{'True Label':<20} | {'Clean Pred':<20} | {'Adv Pred':<20} | {'Clean?':<5} | {'Adv?':<5}")
    print("-" * 90)

    for vid_path in tqdm(selected_videos):
        # Extract True Label from folder name
        true_label = os.path.basename(os.path.dirname(vid_path))
        
        # 1. Test Clean Video
        clean_pred = get_prediction(vid_path)
        
        # 2. Generate and Test Adversarial Video
        adv_path = "temp_adv.mp4"
        try:
            generate_adv_video(vid_path, adv_path)
            adv_pred = get_prediction(adv_path)
        except Exception as e:
            # If generation fails, we can't count this sample for adv stats
            print(f"\nAttack Gen Error: {e}")
            adv_pred = "GenError"

        # 3. Calculate Stats
        # Loose string matching is safer for comparing LLM output
        is_clean_correct = (true_label.lower() in clean_pred.lower()) or (clean_pred.lower() in true_label.lower())
        
        # For adversarial, if it predicts the TRUE label, the defense WORKED (or attack failed).
        # If it predicts something else, the attack succeeded.
        is_adv_correct = (true_label.lower() in adv_pred.lower()) or (adv_pred.lower() in true_label.lower())

        if is_clean_correct: clean_correct += 1
        if is_adv_correct: adv_correct += 1
        total += 1

        print(f"{true_label[:19]:<20} | {clean_pred[:19]:<20} | {adv_pred[:19]:<20} | {str(is_clean_correct):<6} | {str(is_adv_correct):<6}")
        
        # Cleanup
        if os.path.exists(adv_path):
            try:
                os.remove(adv_path)
            except:
                pass

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("BENCHMARK RESULTS")
    print("="*30)
    print(f"Total Videos: {total}")
    if total > 0:
        print(f"Clean Accuracy:       {clean_correct/total*100:.2f}%")
        print(f"Adversarial Accuracy: {adv_correct/total*100:.2f}%")
        # Attack Success Rate is how often the model was WRONG on adv examples
        print(f"Attack Success Rate:  {(1 - (adv_correct/total))*100:.2f}% (approx)")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()