import torch
import numpy as np

class GuidanceSchedulers:
    def __init__(self, timesteps: torch.Tensor, mean_guidance_scale: float):
        self.num_inference_steps = len(timesteps)
        self.timesteps = (timesteps / timesteps.max()).to('cpu').numpy()
        self.mean_guidance_scale = mean_guidance_scale

        timesteps_extended = np.concatenate([self.timesteps, np.array([0.0])])
        self.delta_timesteps = (timesteps_extended[:-1] - timesteps_extended[1:])

    def constant(self):
        return np.full(self.num_inference_steps, self.mean_guidance_scale).tolist()

    def symmetric(self, low: float, high: float, normalize: bool=True):
        half = self.num_inference_steps // 2
        if self.num_inference_steps % 2 == 0:
            up = np.linspace(low, high, half, endpoint=False)
            down = np.linspace(high, low, half, endpoint=True)
            guidance_scales = np.concatenate([up, down])
        else:
            up = np.linspace(low, high, half + 1, endpoint=True)
            down = np.linspace(high, low, half + 1, endpoint=True)[1:]
            guidance_scales = np.concatenate([up, down])

        if normalize:
            guidance_scales = guidance_scales * self.mean_guidance_scale / np.sum(guidance_scales * self.delta_timesteps)
        return guidance_scales.tolist()

    def interval(self, value: float, left: float=0.15, right: float=0.78, normalize: bool=True):
        mask = ((left < self.timesteps) & (self.timesteps < right))
        if normalize:
            value = (self.mean_guidance_scale - self.delta_timesteps[~mask].sum()) / self.delta_timesteps[mask].sum()
        guidance_scales = np.where(mask, value, 1.0)
        return guidance_scales.tolist()

    def stepup(self, low: float, high: float, tau: float=0.25, normalize: bool=True):
        steps = np.linspace(0, 1, self.num_inference_steps)
        guidance_scales = np.where(steps < tau, low, high)
        if normalize:
            guidance_scales = guidance_scales * self.mean_guidance_scale / np.sum(guidance_scales * self.delta_timesteps)
        return guidance_scales.tolist()
    
    def stepdown(self, low: float, high: float, tau: float=0.25, normalize: bool=True):
        steps = np.linspace(0, 1, self.num_inference_steps)
        guidance_scales = np.where(steps < tau, high, low)
        if normalize:
            guidance_scales = guidance_scales * self.mean_guidance_scale / np.sum(guidance_scales * self.delta_timesteps)
        return guidance_scales.tolist()

