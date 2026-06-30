import torch
from PIL import Image
from torchvision import transforms as tfms
import numpy as np
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)

# Note: The model components (vae, unet, etc.) are passed in a dictionary `models`
# to avoid using global variables, making this code modular and server-friendly.

def pil_to_latents(image, models):
    '''Function to convert a single PIL image to latents.'''
    vae = models['diffender_vae']
    transform = tfms.Compose([tfms.ToTensor()])
    init_image = transform(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    with torch.no_grad():
        init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist

def latents_to_pil(latents, models):
    '''Function to convert latents back to a list of PIL images.'''
    vae = models['diffender_vae']
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in images]
    return pil_images

def text_enc(prompts, models, maxlen=None):
    '''Function to encode text prompts.'''
    tokenizer = models['diffender_tokenizer']
    text_encoder = models['diffender_text_encoder']
    if maxlen is None: maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    with torch.no_grad():
        encoded_text = text_encoder(inp.input_ids.to("cuda"))[0].half()
    return encoded_text

def prompt_2_img_i2i_fast_og(prompts, init_img_latent, models, g=7.5, seed=100, strength=0.5, steps=50):
    '''Helper function for generating latents from an initial latent.'''
    unet = models['diffender_unet']
    scheduler = models['diffender_scheduler']

    text = text_enc(prompts, models)
    uncond = text_enc([""], models, text.shape[1])
    emb = torch.cat([uncond, text])

    if seed: torch.manual_seed(seed)
    scheduler.set_timesteps(steps)
    
    init_timestep = int(steps * strength)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device="cuda")

    noise = torch.randn(init_img_latent.shape, device="cuda", dtype=init_img_latent.dtype)
    init_latents = scheduler.add_noise(init_img_latent, noise, timesteps)
    latents = init_latents

    inp = scheduler.scale_model_input(torch.cat([latents] * 2), timesteps)
    
    with torch.no_grad():
        u, t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)

    pred = u + g * (t - u)
    latents = scheduler.step(pred, timesteps, latents).pred_original_sample
    return latents

def create_mask_fast2(init_img_latent, rp, ep, models, n=3, s=0.5):
    '''Creates the difference mask.'''
    diff2 = None
    for idx in range(n):
        empty_noise = prompt_2_img_i2i_fast_og(prompts=ep, init_img_latent=init_img_latent, models=models, strength=s, seed=100 * idx)[0]
        text_noise = prompt_2_img_i2i_fast_og(prompts=rp, init_img_latent=init_img_latent, models=models, strength=s, seed=100 * idx)[0]
        tmp = (text_noise - empty_noise).unsqueeze(0)
        if idx == 0:
            diff2 = tmp
        else:
            diff2 = torch.cat((diff2, tmp), 0)

    mask_t = torch.zeros(diff2[0].shape).to("cuda")
    for idx in range(n):
        mask_t += torch.abs(diff2[idx])
        
    mask_t = torch.mean(mask_t, 0)
    
    mask_max = mask_t.max()
    mask_min = mask_t.min()
    # Normalize mask to [0, 1]
    if mask_max > mask_min:
        mask_t = (mask_t - mask_min) / (mask_max - mask_min)
    
    return mask_t

def improve_mask2(mask):
    '''Applies Gaussian blur to the mask.'''
    mask = mask.unsqueeze(0).unsqueeze(0) # Add batch and channel dimensions
    GaussianBlur = tfms.GaussianBlur((3, 3), sigma=1)
    mask = GaussianBlur(mask)
    # max_pool2d
    kernel_size = 3
    padding = (kernel_size - 1) // 2
    mask = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    return mask.squeeze(0) # Remove channel dimension

def run_diffender(init_img: Image.Image, rp: str, qp: str, models: dict):
    """
    Main function to apply the DIFFender defense to a single PIL image.
    
    Args:
        init_img (PIL.Image.Image): The input image to defend.
        rp (str): The reference prompt (e.g., "a photo").
        qp (str): The query prompt for inpainting (e.g., "a photo").
        models (dict): A dictionary containing all the pre-loaded model components.
    
    Returns:
        PIL.Image.Image: The defended image.
    """
    pipe = models['diffender_pipe']
    
    # Resize image to 512x512, as expected by the model
    init_img_512 = init_img.convert("RGB").resize((512, 512))
    
    # 1. Convert image to initial latents
    init_latents = pil_to_latents(init_img_512, models)

    # 2. Create the mask
    ep = [""] # Empty prompt
    mask_n = create_mask_fast2(init_latents, rp=[rp], ep=ep, models=models, n=3)
    mask = improve_mask2(mask_n)
    
    # 3. Use the inpainting pipeline to reconstruct/defend the image
    # The mask needs to be a PIL image for the pipeline
    mask_pil = tfms.ToPILImage()(mask.cpu())

    with torch.no_grad():
        output_images = pipe(
            prompt=qp,
            image=init_img_512,
            mask_image=mask_pil,
            generator=torch.Generator("cuda").manual_seed(100),
            num_inference_steps=20
        ).images

    # Return the first (and only) defended image, resized back to original size if needed
    defended_image = output_images[0]
    if defended_image.size != init_img.size:
        defended_image = defended_image.resize(init_img.size)
        
    return defended_image