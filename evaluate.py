import os
import torch
from PIL import Image
from tqdm import tqdm
import clip
from torchvision import transforms
import ImageReward as IR
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import argparse
import json

def load_data(prompts_path, image_folder, reference_folder):
    with open(prompts_path, 'r') as f:
        filename_to_prompts = json.load(f)
    gen_images = []
    ref_images = []
    prompts = []

    gen_image_files = [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]
    for filename in tqdm(gen_image_files, desc="Loading generated images"):
        img = Image.open(os.path.join(image_folder, filename)).convert("RGB")
        gen_images.append(img)
        index = os.path.splitext(filename)[0]
        prompts.append(filename_to_prompts[index + '.jpg'])

    ref_image_files = [f for f in os.listdir(reference_folder) if f.endswith(".png") or f.endswith(".jpg")]
    for filename in tqdm(ref_image_files, desc="Loading reference images"):
        img = Image.open(os.path.join(reference_folder, filename)).convert("RGB")
        ref_images.append(img)

    return prompts, gen_images, ref_images

def compute_clip_score(prompts, images, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_inputs = torch.stack([preprocess(img) for img in images]).to(device)
    text_inputs = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        scores = torch.nn.functional.cosine_similarity(image_features, text_features).cpu().numpy().tolist()
    return np.mean(scores)

def compute_imagereward_score(prompts, images, device):
    ir_model = IR.load("ImageReward-v1.0", device=device)
    scores = []
    for prompt, img in tqdm(zip(prompts, images), total=len(prompts), desc="ImageReward"):
        score = ir_model.score(prompt, img)
        scores.append(score)
    return np.mean(scores)

def compute_fid(generated_images, reference_images, device):
    fid = FrechetInceptionDistance().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255).to(torch.uint8)),
    ])

    for img in tqdm(generated_images, desc="FID Generated"):
        tensor_img = preprocess(img).unsqueeze(0).to(device)
        fid.update(tensor_img, real=False)
    for img in tqdm(reference_images, desc="FID Reference"):
        tensor_img = preprocess(img).unsqueeze(0).to(device)
        fid.update(tensor_img, real=True)
    return fid.compute().item()

def compute_saturation(images):
    saturations = []
    for img in tqdm(images, desc="Saturation"):
        hsv = np.array(img.convert('HSV'))
        s = hsv[..., 1] / 255.0
        saturations.append(np.mean(s))
    return np.mean(saturations)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", type=str, default='./data/coco_val_filename_to_prompt.json')
    parser.add_argument("--reference_folder", type=str, default='./data/coco_val_2017')
    parser.add_argument("--image_folder", type=str, default='./imgs/')
    parser.add_argument("--num_inference_steps", type=int, required=True)
    parser.add_argument("--guidance_scheduler", type=str, required=True, choices=['constant', 'symmetric', 'interval', 'stepup', 'stepdown'])
    parser.add_argument("--guidance_method", type=str, required=True, choices=['CFG', 'APG'])
    parser.add_argument("--guidance_scale", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    prompts_path = args.prompts_path
    reference_folder = args.reference_folder
    image_folder = args.image_folder
    num_inference_steps = args.num_inference_steps
    guidance_scheduler = args.guidance_scheduler
    guidance_method = args.guidance_method
    guidance_scale = args.guidance_scale
    device = args.device

    subfolder_name = f"NFE{num_inference_steps}_gs{guidance_scale}_{guidance_method}_{guidance_scheduler}"
    image_subfolder = os.path.join(image_folder, subfolder_name)
    if not os.path.exists(image_subfolder):
        raise ValueError(f"Image folder {image_subfolder} does not exist.")
    prompts, gen_images, ref_images = load_data(prompts_path, image_subfolder, reference_folder)

    print("Calculating CLIP alignment...")
    clip_scores = compute_clip_score(prompts, gen_images, device)
    print(f"CLIP Score: {clip_scores:.3f}")

    print("Calculating ImageReward...")
    imagereward_scores = compute_imagereward_score(prompts, gen_images, device)
    print(f"ImageReward: {imagereward_scores:.3f}")

    print("Calculating FID...")
    fid_score = compute_fid(gen_images, ref_images, device)
    print(f"FID: {fid_score:.3f}")

    print("Calculating Saturation...")
    saturation_scores = compute_saturation(gen_images)
    print(f"Saturation: {saturation_scores:.3f}")

    print("\nSummary:")
    print(f"Evaluated on {subfolder_name}")
    print(f"CLIP Score: {clip_scores:.3f}")
    print(f"ImageReward: {imagereward_scores:.3f}")
    print(f"FID: {fid_score:.3f}")
    print(f"Saturation: {saturation_scores:.3f}")

if __name__ == "__main__":
    main()
