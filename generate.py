import os
import json
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from tvpipelines import TVSD3Pipeline
from guidance_schedulers import GuidanceSchedulers

def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", type=str, default='./data/coco_val_filename_to_prompt.json')
    parser.add_argument("--num_inference_steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, default='./imgs/')
    parser.add_argument("--guidance_scheduler", type=str, required=True, choices=['constant', 'symmetric', 'interval', 'stepup', 'stepdown'])
    parser.add_argument("--guidance_method", type=str, required=True, choices=['CFG', 'APG'])
    parser.add_argument("--guidance_scale", type=float, required=True)
    args = parser.parse_args()

    prompts_path = args.prompts_path
    num_inference_steps = args.num_inference_steps
    seed = args.seed
    output_folder = args.output_folder
    guidance_scheduler = args.guidance_scheduler
    guidance_method = args.guidance_method
    guidance_scale = args.guidance_scale

    rank, world_size, device_id = setup_distributed()
    device = f"cuda:{device_id}"

    output_subfolder = os.path.join(output_folder, f"NFE{num_inference_steps}_gs{guidance_scale}_{guidance_method}_{guidance_scheduler}")
    if rank == 0:
        os.makedirs(output_subfolder, exist_ok=True)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            all_prompts = json.load(f)
        prompts_to_process = list(all_prompts.items())
    dist.barrier()

    broadcast_list = [prompts_to_process] if rank == 0 else [None]
    dist.broadcast_object_list(broadcast_list, src=0)
    prompts_to_process = broadcast_list[0]
    prompts_for_this_rank = prompts_to_process[rank::world_size]

    if rank == 0:
        print("Initializing Pipeline...")
    pipe = TVSD3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.set_guidance_method(guidance_method)
    pipe.warmup(num_inference_steps=num_inference_steps)
    guidance_schedulers = GuidanceSchedulers(pipe.scheduler.timesteps, mean_guidance_scale=guidance_scale)
    if guidance_scheduler == 'constant':
        guidance_scales = guidance_schedulers.constant()
    elif guidance_scheduler == 'symmetric':
        guidance_scales = guidance_schedulers.symmetric(low=1.0, high=2*guidance_scale-1.0)
    elif guidance_scheduler == 'interval':
        guidance_scales = guidance_schedulers.interval(value=guidance_scale)
    elif guidance_scheduler == 'stepup':
        guidance_scales = guidance_schedulers.stepup(low=guidance_scale/3, high=guidance_scale, normalize=False)
    elif guidance_scheduler == 'stepdown':
        guidance_scales = guidance_schedulers.stepdown(low=guidance_scale/3, high=guidance_scale, normalize=False)
    else:
        raise ValueError(f"Unknown guidance scheduler: {guidance_scheduler}")

    if rank == 0:
        print(f"Starting generation...")
    generator = torch.Generator(device=device).manual_seed(seed)
    for key, prompt in tqdm(prompts_for_this_rank, desc=f"GPU Rank {rank}", position=rank, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type='pil',
            generator=generator,
            guidance_scale=guidance_scales
        ).images[0]
        base_filename = os.path.splitext(key)[0]
        output_filename = os.path.join(output_subfolder, f"{base_filename}.png")
        image.save(output_filename)

    dist.destroy_process_group()
    if rank == 0:
        print("\nAll tasks are done!")

if __name__ == "__main__":
    main()