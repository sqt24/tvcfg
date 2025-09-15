import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from typing import List, Union, Optional, Callable, Dict, Any
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps, calculate_shift
import torch.nn.functional as F

def call_cos_tensor(tensor1, tensor2):
    tensor1 = tensor1/torch.linalg.norm(tensor1,dim=1,keepdim=True)
    tensor2 = tensor2/torch.linalg.norm(tensor2,dim=1,keepdim=True)
    cosvalue = torch.sum(tensor1*tensor2,dim=1,keepdim=True)
    return cosvalue

def compute_perpendicular_component(latent_diff, latent_hat_uncond):
    shape = latent_diff.size()
    latent_diff = latent_diff.view(latent_diff.size(0), -1).float()
    latent_hat_uncond = latent_hat_uncond.view(latent_hat_uncond.size(0), -1).float()
    
    if latent_diff.size() != latent_hat_uncond.size():
        raise ValueError("latent_diff and latent_hat_uncond must have the same shape [n, d].")
    
    dot_product = torch.sum(latent_diff * latent_hat_uncond, dim=1, keepdim=True)  # [n, 1]
    norm_square = torch.sum(latent_hat_uncond * latent_hat_uncond, dim=1, keepdim=True)  # [n, 1]
    projection = (dot_product / (norm_square + 1e-8)) * latent_hat_uncond
    perpendicular_component = latent_diff - projection
    
    return projection.view(shape),perpendicular_component.view(shape)

class TVSD3Pipeline(StableDiffusion3Pipeline):
    def _post_init(self):
        self._guidance_method = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        pipe = super().from_pretrained(*args, **kwargs)
        pipe._post_init()
        return pipe
    
    @property
    def do_classifier_free_guidance(self):
        return True

    def set_guidance_method(self, method: str):
        legal_methods = ('CFG', 'APG')
        if method not in legal_methods:
            raise ValueError(f"Unsupported guidance method: {method}. Supported methods are {legal_methods}.")
        self._guidance_method = method

    def warmup(self, num_inference_steps: int):
        self("initializing...", num_inference_steps=num_inference_steps, guidance_scale=1.5, output_type='pil')

    @torch.no_grad()
    def __call__(self, *args, **kwargs) -> StableDiffusion3PipelineOutput:
        if self._guidance_method is None:
            raise RuntimeError("Guidance method not set. Please call set_guidance_method() first.")
        return getattr(self, self._guidance_method)(*args, **kwargs)

    @torch.no_grad()
    def CFG(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Union[float, List[float], np.ndarray] = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # --- deal with per-step guidance_scale ---
        if isinstance(guidance_scale, float):
            guidance_scale = [guidance_scale] * num_inference_steps
        elif isinstance(guidance_scale, np.ndarray):
            guidance_scale = guidance_scale.tolist()
        elif isinstance(guidance_scale, list):
            pass
        else:
            raise ValueError(f"guidance_scale should be a float, list of floats or numpy array, but got {type(guidance_scale)}.")
        if len(guidance_scale) != num_inference_steps:
                raise ValueError(
                    f"guidance_scale should be a single float or a list of floats with length {num_inference_steps}, but got {len(guidance_scale)}."
                )
        # --- modification finished ---
        
        self.check_inputs(
            prompt, prompt_2, prompt_3, height, width, negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2, negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None)
        
        (
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3, negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2, negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, device=device,
            clip_skip=self.clip_skip, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length, lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (width // self.transformer.config.patch_size)
            mu = calculate_shift(
                image_seq_len, self.scheduler.config.base_image_seq_len, self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift, self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs,
        )
        self._num_timesteps = len(timesteps)

        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, ip_adapter_image_embeds, device,
                batch_size * num_images_per_prompt, self.do_classifier_free_guidance,
            )
            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds, joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # --- use self.guidance_scale[i] ---
                    noise_pred = noise_pred_uncond + self._guidance_scale[i] * (noise_pred_text - noise_pred_uncond)

                    should_skip_layers = (True if i > num_inference_steps * skip_layer_guidance_start and i < num_inference_steps * skip_layer_guidance_stop else False)
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds, joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False, skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds)
                
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)

    @torch.no_grad()
    def APG(
        self,
        prompt: Union[str, List[str]] = None,
        eta = 0,
        beta = 0,
        r = 10000,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float], np.ndarray] = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # --- deal with per-step guidance_scale ---
        if isinstance(guidance_scale, float):
            guidance_scale = [guidance_scale] * num_inference_steps
        elif isinstance(guidance_scale, np.ndarray):
            guidance_scale = guidance_scale.tolist()
        elif isinstance(guidance_scale, list):
            pass
        else:
            raise ValueError(f"guidance_scale should be a float, list of floats or numpy array, but got {type(guidance_scale)}.")
        if len(guidance_scale) != num_inference_steps:
                raise ValueError(
                    f"guidance_scale should be a single float or a list of floats with length {num_inference_steps}, but got {len(guidance_scale)}."
                )
        # --- modification finished ---

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=True,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if True:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            class MomentumBuffer:
                def __init__(self, momentum=0.9):
                    self.momentum = momentum
                    self.running_average = None

                def update(self, new_value):
                    if self.running_average is None:
                        self.running_average = new_value
                    else:
                        self.running_average = self.momentum * self.running_average + (1 - self.momentum) * new_value


            def project(
                v0: torch.Tensor,  # [B, C, H, W]
                v1: torch.Tensor,  # [B, C, H, W]
            ):
                dtype = v0.dtype
                v0, v1 = v0.double(), v1.double()
                v1 = F.normalize(v1, dim=[-1, -2, -3])  # normalize over all elements
                v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
                v0_orthogonal = v0 - v0_parallel
                return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

            def normalized_guidance(
                pred_cond: torch.Tensor,     # [B, C, H, W]
                pred_uncond: torch.Tensor,   # [B, C, H, W]
                guidance_scale: float,
                momentum_buffer: MomentumBuffer = None,
                eta: float = 1.0,
                norm_threshold: float = 0.0,
            ):
                diff = pred_cond - pred_uncond

                if momentum_buffer is not None:
                    momentum_buffer.update(diff)
                    diff = momentum_buffer.running_average

                if norm_threshold > 0:
                    ones = torch.ones_like(diff)
                    diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                    #print(diff_norm)
                    scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
                    diff = diff * scale_factor

                diff_parallel, diff_orthogonal = project(diff, pred_cond)
                normalized_update = diff_orthogonal + eta * diff_parallel
                pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
                return pred_guided
            momentum_buffer = MomentumBuffer(momentum=beta)
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if True else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if True:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    weight = self.guidance_scale[i]
                    n, c, w, h = noise_pred_text.shape
                    if self.scheduler.step_index == None:
                        sigma = self.scheduler.sigmas[0]
                    else:
                        sigma = self.scheduler.sigmas[self.scheduler.step_index]
                    #print(sigma)
                    latent_hat_text = latents - sigma * noise_pred_text
                    latent_hat_uncond = latents - sigma * noise_pred_uncond


                    latent_new = normalized_guidance(
                        latent_hat_text,
                        latent_hat_uncond,
                        weight,
                        momentum_buffer=momentum_buffer,
                        eta=eta,
                        norm_threshold=r)


                    noise_pred = (latents - latent_new)/sigma
                    noise_pred = noise_pred.view(n, c, w, h).to(latents.dtype)



                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: [MPS] Add support for autocast in MPS  by kulinseth · Pull Request #99272 · pytorch/pytorch
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
