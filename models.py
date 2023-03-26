import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# APACHE 2.0
# - https://github.com/christophschuhmann/improved-aesthetic-predictor
# - - reused trained model
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1))
    def forward(self, x):
        return self.layers(x)

# =============================================================================
# Misc
# =============================================================================
import torchvision.transforms.functional as FT

#normalize tensor
def normalized_pt(b, axis=-1):
    l2 = torch.norm(b)
    l2[l2 == 0] = 1
    return b / l2.reshape(1,1)

#get transforms to try
def get_transform_to_params(mode='soft'):
    if mode == 'hard':
        a = 0.5
        transform_to_params = {}
        transform_to_params[FT.adjust_brightness] = {'brightness_factor':((a, 1/a),)}
        transform_to_params[FT.adjust_contrast] = {'contrast_factor':((a, 1/a),)}
        transform_to_params[FT.adjust_gamma] = {'gamma':((a, 1/a),), 'gain':((0.9,1.1),)}
        transform_to_params[FT.adjust_hue] = {'hue_factor':((-0.1, 0.1),)}
        transform_to_params[FT.invert] = {}
        transform_to_params[FT.solarize] = {'threshold':((0, 1),)}
        transform_to_params[FT.posterize] = {'bits': (np.random.randint,(0,9))}
        transform_to_params[FT.adjust_saturation] = {'saturation_factor':((a,1/a),)}
        transform_to_params[FT.adjust_sharpness] = {'sharpness_factor':((a, 1/a),)}
        transform_to_params[FT.autocontrast] = {}
        # transform_to_params[FT.gaussian_blur] = {'kernel_size': (np.random.randint,(1, 5))}
        transform_to_params[FT.equalize] = {}
        print('|- HARD MODE')
    elif mode == 'soft':
        a = 0.9
        transform_to_params = {}
        transform_to_params[FT.adjust_brightness] = {'brightness_factor':((a, 1/a),)}
        transform_to_params[FT.adjust_contrast] = {'contrast_factor':((a, 1/a),)}
        transform_to_params[FT.adjust_gamma] = {'gamma':((a, 1/a),), 'gain':((0.9,1.1),)}
        transform_to_params[FT.adjust_saturation] = {'saturation_factor':((a,1/a),)}
        transform_to_params[FT.adjust_sharpness] = {'sharpness_factor':((a, 1/a),)}
        transform_to_params[FT.autocontrast] = {}
        transform_to_params[FT.gaussian_blur] = {'kernel_size': (np.random.randint,(1, 3))}
        transform_to_params[FT.equalize] = {}
        print('|- SOFT MODE')
    return transform_to_params


























