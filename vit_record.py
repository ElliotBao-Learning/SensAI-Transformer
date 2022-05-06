from matplotlib.pyplot import axis
import numpy as np
import torch
import contextlib
import torch.nn.functional as F
from collections import OrderedDict 
from typing import Dict, Callable

def avg_scoring(activation):
    """
    Calculate the Average Percentage of Zeros Score of the feature map activation layer output
    """
    if activation.dim() == 3:
        featuremap_apoz_mat = activation.mean(dim=(0, 1))
    else:
        raise ValueError(
            f"activation_channels_avg: Unsupported shape: {activation.shape}")
    return featuremap_apoz_mat.mul(100).cpu()


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)

class ViTRecord:
    def __init__(self, model, arch):
        self.scores = []
        self.num_batches = 0
        self.layer_idx = 0
        self._candidates_by_layer = None
        self._model = model
        # switch to evaluate mode
        self._model.eval()
        self._model.apply(lambda m: m.register_forward_hook(self._hook))
        self.arch = arch

    def parse_activation(self, feature_map):
        score = feature_map.cpu().numpy()

        if self.num_batches == 0:
            self.scores.append(score)
        else:
            self.scores[self.layer_idx] += score
        self.layer_idx += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        temp_scores = [0]*(len(self.scores)//2)
        for i, score in enumerate(self.scores):
            if i%4 == 0:
                temp_scores[i//2] = score.mean(axis=(0, 1))
                temp_scores[i//2+1] = (self.scores[i+1].mean(axis=(0, 1)) + self.scores[i+2].mean(axis=(0, 1)) + self.scores[i+3].mean(axis=(0, 1)))/3
        self.scores = np.array(temp_scores)
        remove_all_forward_hooks(self._model)



    def record_batch(self, *args, **kwargs):
        # reset layer index
        self.layer_idx = 0
        with torch.no_grad():
            # output is not used
            _ = self._model(*args, **kwargs)
        self.num_batches += 1

    def _hook(self, module, input, output):
        """Apply a hook to RelU layer"""
        if module.__class__.__name__ == 'channel_selection':
            self.parse_activation(output)
