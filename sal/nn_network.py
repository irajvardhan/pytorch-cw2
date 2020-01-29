import os
import numpy as np
import copy
import torch
from torchvision.models.resnet import ResNet
from torch.nn import functional as F
from .utils.pytorch_fixes import *
from torch.nn import Module
class SaliencyModel(Module):
    def __init__(self, encoder, channel_dims): #encoder_scales, encoder_base, upsampler_scales, upsampler_base,
        super(SaliencyModel, self).__init__()
        #assert upsampler_scales <= encoder_scales

        encoder.eval()
        self.encoder = encoder
        self.channel_dims=channel_dims  
        self.upsampler_scales = len(channel_dims)-2#upsampler_scales
        self.encoder_scales = len(channel_dims)#encoder_scales
        down = self.encoder_scales
        modulator_sizes = []
        out_chans = 0
        for up in reversed(range(self.upsampler_scales)):
            #upsampler_chans = upsampler_base * 2**(up+1)
            inc = channel_dims[up+2] if down==(self.encoder_scales) else out_chans #upsampler_chans
            out_chans = int(channel_dims[up+1]/4)
            modulator_sizes.append(inc)
            self.add_module('up%d'%up,
                            UNetUpsampler(
                                in_channels=inc,
                                passthrough_channels=channel_dims[up+1],#int(encoder_chans/2),
                                out_channels=(out_chans),
                                follow_up_residual_blocks=1,
                                activation_fn=lambda: nn.ReLU(),
                            ))
            
            down -= 1
        self.to_saliency_chans = nn.Conv2d(in_channels=out_chans, out_channels=1, kernel_size=1)
        
    def get_trainable_parameters(self):
        all_params = self.parameters()
        if not self.fix_encoder: return set(all_params)
        unwanted = self.encoder.parameters()
        return set(all_params) - set(unwanted) - (set(self.selector_module.parameters()) if self.allow_selector else set([]))

    def forward(self, inputs):     
        num_channels=self.channel_dims[-1]#2048
        out = self.encoder(inputs)
        dim_chans=inputs.size()[-1]
        down = self.encoder_scales
        main_flow = out[down]
        for up in reversed(range(self.upsampler_scales)):
            assert down > 0
            main_flow = self._modules['up%d'%up](main_flow, out[down-1])
            down -= 1

        outputs = self.to_saliency_chans(main_flow)
        outputs = F.upsample(outputs,(dim_chans, dim_chans), mode='bilinear')
        return outputs 