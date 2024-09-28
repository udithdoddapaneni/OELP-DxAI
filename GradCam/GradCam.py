import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cv2 import resize, addWeighted
from typing import Callable
from torch import nn
from PIL import Image
from torchvision.transforms import Normalize

class GradCAM():
    def __init__(self, Model: nn.Module, LayerName: str, Normalization: Normalize) -> None:
        def trigger(obj, attr_name):
            class wrapper(getattr(obj, attr_name).__class__):
                def __call__(this, *args, **kwargs):
                    activations:torch.Tensor = super().__call__(*args, **kwargs)
                    activations.register_hook(self.register_gradient)
                    self.register_activations(activations)
                    return activations
            new_attr = wrapper()
            new_attr.__dict__ = getattr(obj, attr_name).__dict__
            setattr(obj, attr_name, new_attr)
            return new_attr
    
        Layer = trigger(Model, LayerName)
        self.Model = Model
        self.Layer = Layer
        self.gradient = None
        self.activations = None
        self.Normalization = Normalization
        self.heatmap: Image.Image|None = None
        self.device = torch.device('cpu')

        self.Model.to(self.device)

    def register_gradient(self, grad: torch.Tensor) -> None:
        self.gradient = grad
    
    def register_activations(self, activations: torch.Tensor) -> None:
        self.activations = activations

    def get_activation_gradients(self) -> torch.Tensor:
        return self.gradient
    
    def get_activations(self, im: torch.Tensor) -> torch.Tensor:
        self.Model(im)
        return self.activations
        
    def gradCAM_heatmap(self, preprocessor: Callable, im: Image.Image | np.ndarray | torch.Tensor, label: int, final_shape: tuple = (512,512)) -> Image.Image:
        if final_shape[0] != final_shape[1]:
            raise Exception('final dimensions must be equal')
        self.Model.eval()
        image = im
        im: torch.Tensor = preprocessor(im)
        im = im.to(self.device)
        
        activations = self.get_activations(im)
        output: torch.Tensor = self.Model(im)
        output[0][label].backward()
        activation_gradients = self.get_activation_gradients()
        averaged_gradients = torch.mean(activation_gradients, dim=(0,2,3))
        for i in range(len(averaged_gradients)):
            activations[:, i, :, :] *= averaged_gradients[i]
        heatmap = torch.mean(activations, dim=(0,1))
        heatmap = heatmap/heatmap.max()
        heatmap = np.array(resize(heatmap.detach().numpy(), final_shape))

        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        heatmap_colored = np.uint8(255*heatmap_colored)

        image = np.array(resize(image, final_shape))
        alpha = 0.4
        blended_image = addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)
        self.heatmap = Image.fromarray(blended_image)
        return self.heatmap
    
    def showHeatMap(self):
        self.heatmap.show()


