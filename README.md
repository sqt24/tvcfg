# Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models

## Installation

1. Clone this repo.

```
git clone https://github.com/sqt24/tvcfg.git
cd tvcfg
```

2. Create the environment.

```
conda create -n tvcfg python=3.11 -y
conda activate tvcfg
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
pip install --prefer-binary -r requirements.txt
```

3. Download dataset.

```
mkdir -p data/coco_val_2017
wget -c http://images.cocodataset.org/zips/val2017.zip -O data/coco_val_2017/val2017.zip
unzip -q -j data/coco_val_2017/val2017.zip 'val2017/*.jpg' -d data/coco_val_2017
rm data/coco_val_2017/val2017.zip
```

## Usage

### Generation

To generate images, using the command

```
torchrun --nproc_per_node=<NUM_GPUS> \
    generate.py \
    --num_inference_steps <NFE> \
    --guidance_scheduler <GUIDANCE_SCHEDULER> \
    --guidance_method <GUIDANCE_METHOD> \
    --guidance_scale <OMEGA>
```

* `nproc_per_node`: Number of processes to launch on this node; typically equals the number of available GPUs.

* `num_inference_steps`: Number of inference steps, also referred to as NFE.

* `guidance_scheduler`: Time profile of the guidance scale. 

  |  Choice   |    Description    |
  | :-------: | :---------------: |
  | constant  |    vanilla-CFG    |
  | symmetric |      TV-CFG       |
  | interval  |   interval-CFG    |
  |  stepup   | early low weight  |
  | stepdown  | early high weight |


* `guidance_method`: Guidance method, choosing between `CFG` and `APG`. 

* `guidance_scale`: Guidance scale $\omega$. 

### Evaluation

To evaluate the generated images, using the command

```
python evaluate.py \
    --num_inference_steps <NFE> \
    --guidance_scheduler <GUIDANCE_SCHEDULER> \
    --guidance_method <GUIDANCE_METHOD> \
    --guidance_scale <OMEGA>
```

which will compute the IR, CLIP, FID, and Saturation metrics of the generated results under the corresponding parameters (Generation is required beforehand).

### Visualization

To visualize the generation diversity, using the command

```
torchrun --nproc_per_node=<NUM_GPUS> \
    visualize.py \
    --prompt <PROMPT> \
    --num_inference_steps <NFE> \
    --guidance_scheduler <GUIDANCE_SCHEDULER> \
    --guidance_method <GUIDANCE_METHOD> \
    --guidance_scale <OMEGA>
```

which will sample 16 images under the \<PROMPT\> condition to visually compare the diversity of the model’s generated results.

