import os
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from tvpipelines import TVSD3Pipeline
from guidance_schedulers import GuidanceSchedulers
from PIL import Image

def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A view of a bathroom that is clean.")
    parser.add_argument("--num_inference_steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, default='./visualizations/')
    parser.add_argument("--guidance_scheduler", type=str, required=True, choices=['constant', 'symmetric', 'interval', 'stepup', 'stepdown'])
    parser.add_argument("--guidance_method", type=str, required=True, choices=['CFG', 'APG'])
    parser.add_argument("--guidance_scale", type=float, required=True)
    parser.add_argument("--num_pics", type=int, default=16)
    parser.add_argument("--grid_pic_size", type=int, default=128)
    args = parser.parse_args()

    prompt = args.prompt
    num_inference_steps = args.num_inference_steps
    seed = args.seed
    output_folder = args.output_folder
    guidance_scheduler = args.guidance_scheduler
    guidance_method = args.guidance_method
    guidance_scale = args.guidance_scale
    num_pics = args.num_pics
    grid_pic_size = args.grid_pic_size

    rank, world_size, device_id = setup_distributed()
    device = f"cuda:{device_id}"

    output_subfolder = os.path.join(output_folder, prompt.replace(' ', '_'))
    if rank == 0:
        os.makedirs(output_subfolder, exist_ok=True)
        seeds = [seed + i for i in range(num_pics)]
    dist.barrier()

    broadcast_list = [seeds] if rank == 0 else [None]
    dist.broadcast_object_list(broadcast_list, src=0)
    seeds = broadcast_list[0]
    seeds_for_this_rank = seeds[rank::world_size]

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
    images = []
    for seed in tqdm(seeds_for_this_rank, desc=f"GPU Rank {rank}", position=rank, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type='pil',
            generator=torch.Generator(device=device).manual_seed(seed),
            guidance_scale=guidance_scales
        ).images[0]
        images.append(image)

    if rank == 0:
        gathered = [None for _ in range(world_size)]
        dist.gather_object(images, gathered, dst=0)
        merged = [item for sublist in gathered for item in sublist]

        side = int(num_pics ** 0.5)
        thumb_size = (grid_pic_size, grid_pic_size)
        canvas = Image.new('RGB', (side * grid_pic_size, side * grid_pic_size))

        for i, img in enumerate(merged):
            img_resized = img.resize(thumb_size)
            x = (i % side) * grid_pic_size
            y = (i // side) * grid_pic_size
            canvas.paste(img_resized, (x, y))
        fullname = f"NFE{num_inference_steps}_gs{guidance_scale}_{guidance_method}_{guidance_scheduler}"
        canvas.save(os.path.join(output_subfolder, f"{fullname}.png"))
    else:
        dist.gather_object(images, dst=0)

    dist.destroy_process_group()
    if rank == 0:
        print("\nAll tasks are done!")

if __name__ == "__main__":
    main()