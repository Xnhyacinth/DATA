import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F

import torch.nn as nn


class DATAInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2=64):
        super().__init__()

        self.linear_data = nn.Linear(in_features, out_features, bias)
        self.data_down = nn.Linear(in_features, r, bias=False)
        self.data_up = nn.Linear(r, out_features, bias=False)
        self.data_down2 = nn.Linear(in_features, r2, bias=False)
        self.data_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale1 = 1.0
        self.scale2 = 1.0
        self.scale3 = 1.0
        
        nn.init.normal_(self.data_down.weight, std=1 / r**2)
        nn.init.zeros_(self.data_up.weight)

        nn.init.normal_(self.data_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.data_up2.weight)

   
    
    def forward(self, input):
        # try:
        #     x = self.linear_data(input) + self.data_up(self.data_down(input)) * self.scale1 + self.data_up2(self.data_down2(input)) * self.scale2
        # except:
        #     breakpoint()
        return self.linear_data(input) * self.scale3 + self.data_up(self.data_down(input)) * self.scale1 + self.data_up2(self.data_down2(input)) * self.scale2

def uninject_trainable_data(model: nn.Module, target_replace_module: List[str] = ["CrossAttention", "Attention"]):
    """
    Uninject DATA from the model, restoring the original nn.Linear modules.
    """

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                # Check if the child module is an instance of DATAInjectedLinear
                if isinstance(_child_module, DATAInjectedLinear):
                    # Extract the original linear_data layer
                    original_linear = _child_module.linear_data

                    # Replace DATAInjectedLinear with the original nn.Linear module
                    _module._modules[name] = original_linear


def inject_trainable_data(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 4,
    r2: int = 16
):
    """
    inject data into model, and returns data parameter groups.
    """

    require_grad_params = []
    names = []

    for layer in model.modules():
        for _, param in layer.named_parameters():
            param.requires_grad = False
    
    for _module in model.modules():
        # for _, param in _module.named_parameters():
        #     breakpoint()
        if _module.__class__.__name__ in target_replace_module:
            
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = DATAInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2
                    )
                    _tmp.linear_data.weight = weight
                    if bias is not None:
                        _tmp.linear_data.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[name].data_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].data_down.parameters())
                    )
                    _module._modules[name].data_up.weight.requires_grad = True
                    _module._modules[name].data_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[name].data_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].data_down2.parameters())
                    )
                    _module._modules[name].data_up2.weight.requires_grad = True
                    _module._modules[name].data_down2.weight.requires_grad = True
                    names.append(name)
    
    return require_grad_params, names
