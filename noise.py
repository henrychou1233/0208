import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class AdaptiveNoiseGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config.model.device
        self.noise_history = []
        self.adaptation_rate = 0.01
        self.min_std = 0.01
        self.max_std = 2.0

    def calculate_adaptive_std(self, x):
        if len(self.noise_history) > 0:
            recent_noise = torch.stack(self.noise_history[-100:])
            current_std = torch.std(recent_noise)
            signal_std = torch.std(x)
            
            adaptive_std = torch.clamp(
                signal_std * self.adaptation_rate,
                self.min_std,
                self.max_std
            )
            return adaptive_std
        return 1.0

    def get_noise(self, x, noise_type='adaptive_gaussian', seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if noise_type == 'adaptive_gaussian':
            std = self.calculate_adaptive_std(x)
            noise = torch.randn_like(x).to(self.device) * std
            
        elif noise_type == 'gaussian':
            noise = torch.randn_like(x).to(self.device)
            
        elif noise_type == 'uniform':
            noise = torch.rand_like(x).to(self.device) * 2 - 1
            
        elif noise_type == 'laplace':
            loc = torch.zeros_like(x)
            scale = torch.ones_like(x)
            noise = torch.distributions.Laplace(loc, scale).sample().to(self.device)
            
        elif noise_type == 'mixture':
            noise1 = torch.randn_like(x).to(self.device)
            noise2 = torch.randn_like(x).to(self.device) * 2
            mix = torch.rand_like(x).to(self.device)
            noise = torch.where(mix > 0.5, noise1, noise2)
            
        elif noise_type == 'perlin':
            noise = self.generate_perlin_noise(x.shape)
            
        else:
            print(f'Unknown noise type: {noise_type}. Using default Gaussian noise.')
            noise = torch.randn_like(x).to(self.device)

        if len(self.noise_history) >= 100:
            self.noise_history.pop(0)
        self.noise_history.append(noise.detach())

        return noise

    def generate_perlin_noise(self, shape):
        def fade(t): return 6 * t**5 - 15 * t**4 + 10 * t**3
        def lerp(a, b, x): return a + x * (b - a)
        def grad(hash, x):
            h = hash & 15
            grad = 1 + (h & 7)
            if h & 8: grad = -grad
            return grad * x

        def normalized_noise(shape):
            noise = torch.zeros(shape).to(self.device)
            p = torch.randperm(256).to(self.device)
            p = torch.cat([p, p])
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    x, y = i / shape[0], j / shape[1]
                    X, Y = int(x) & 255, int(y) & 255
                    x, y = x - int(x), y - int(y)
                    u, v = fade(x), fade(y)
                    A = (p[X] + Y) & 255
                    B = (p[X + 1] + Y) & 255
                    
                    noise[i, j] = lerp(
                        lerp(grad(p[A], x), grad(p[B], x-1), u),
                        lerp(grad(p[A + 1], x), grad(p[B + 1], x-1), u),
                        v
                    )
            
            noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
            return noise

        base_noise = normalized_noise((shape[-2], shape[-1]))
        noise = base_noise.expand(shape)
        return noise

def get_noise(x, config, noise_type='adaptive_gaussian', seed=None):
    noise_generator = AdaptiveNoiseGenerator(config)
    return noise_generator.get_noise(x, noise_type=noise_type, seed=seed)