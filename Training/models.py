import torch
import torch.nn as nn
import os


class DnCNN_RL(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_RL, self).__init__()

        self.dncnn = DnCNN(channels=channels, num_of_layers=num_of_layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return noise



class BUIFD(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(BUIFD, self).__init__()

        self.dncnn = DnCNN(channels=channels, num_of_layers=num_of_layers)

        self.noisecnn = NoiseCNN(channels=channels)
        self.FinalFusionLayers = FinalFusionLayers(channels=channels)

    def forward(self, x):
        noisy_input = x
        
        # PRIOR:
        noise = self.dncnn(x)
        prior = noisy_input - noise
        
        # NOISE LVL:
        noise_level = self.noisecnn(x)

        # FUSION:
        denoised_image = self.FinalFusionLayers(noisy_input, prior, noise_level)
        noise_out = noisy_input - denoised_image

        return noise_out, noise_level








class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out



class NoiseCNN(nn.Module):
    def __init__(self, channels, num_of_layers=5):
        super(NoiseCNN, self).__init__()
        kernel_size = 5
        padding = 2
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.noisecnn = nn.Sequential(*layers)
        self.sigmoid_mapping = nn.Sigmoid()

    def forward(self, x):
        noise_level = self.noisecnn(x)
        noise_level = self.sigmoid_mapping(noise_level)

        return noise_level



class FinalFusionLayers(nn.Module):
    def __init__(self, channels):
        super(FinalFusionLayers, self).__init__()
        kernel_size = 3
        padding = 1
        features = 16
        dilation = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=5*channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation))
        
        self.fusion_layers = nn.Sequential(*layers)

    def forward(self, a, b, c):
        noisy_input = a
        prior = b
        noise_level = c

        channel_0 = noisy_input
        channel_1 = prior
        channel_2 = noise_level
        channel_3 = noisy_input * (1-noise_level)
        channel_4 = prior * noise_level

        x = torch.cat((channel_0, channel_1, channel_2, channel_3, channel_4), 1)
        fused_out = self.fusion_layers(x)
        
        return fused_out

