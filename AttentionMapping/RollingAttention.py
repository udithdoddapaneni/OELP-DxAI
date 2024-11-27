from typing import Iterable
from torchvision.transforms import Resize
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def rolling_attention_map(attentions:Iterable[torch.Tensor], size:tuple[int] = (224,224), layers:int|None = None) -> np.ndarray:
    rolling_attention = torch.eye(attentions[0].shape[-1]).to(attentions[0].device) # shape: tokens, tokens
    if layers == None:
        layers = len(attentions)
    for idx, attention in enumerate(attentions):
        if idx == layers:
            break
        avg_attention = attention.mean(dim=1).squeeze(0) # shape: tokens, tokens
        avg_attention += torch.eye(avg_attention.shape[-1]) # shape: tokens, tokens
        avg_attention /= avg_attention.sum() # normalization
        rolling_attention = rolling_attention @ avg_attention
        # rolling_attention = avg_attention @ rolling_attention
        
    rolling_attention = rolling_attention.detach().numpy()
    cls_attention = rolling_attention[0, 1:]
    cls_attention = cls_attention.reshape(int(np.sqrt(cls_attention.shape[0])), int(np.sqrt(cls_attention.shape[0])))
    resize = Resize(size)
    cls_attention = resize(torch.tensor(cls_attention).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())
    return cls_attention

# Image.Image is PIL.Image.Image, just incase if you face import errors @dhruv
def rolling_attention_heatmap(attentions:Iterable[torch.Tensor], image: Image.Image, size:tuple[int] = (224,224), layers=None) -> None:
    attention_map = rolling_attention_map(attentions=attentions, size=size, layers=layers)
    resize = Resize(size)
    image = resize(image)
    plt.imshow(image)
    plt.imshow(attention_map, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Attention Intensity')
    plt.show()

def get_rolling_heatmaps_from_model(feature_extractor, model, image: Image.Image, size:tuple[int] = (224, 224), layers=None) -> None:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True, return_dict=True)
    attentions = outputs.attentions
    print("output class probabilities:")
    print(nn.functional.softmax(outputs.logits[0], dim=0).detach().numpy())
    rolling_attention_heatmap(attentions=attentions, image=image, size=size, layers=layers)

def rolling_heatmaps_all_layer(feature_extractor, model, image: Image.Image, size:tuple[int] = (224, 224)) -> None:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True, return_dict=True)
    attentions = outputs.attentions
    print("output class probabilities:")
    print(nn.functional.softmax(outputs.logits[0], dim=0).detach().numpy())
    
    fig, ax = plt.subplots(1, len(attentions)+1, figsize=(20,20))
    resize = Resize(size)
    image = resize(image)
    ax[0].imshow(image)
    ax[0].set_title("original image")
    for layer in range(len(attentions)):
        attention_map = rolling_attention_map(attentions, size, layer+1)
        ax[layer+1].imshow(image)
        ax[layer+1].imshow(attention_map, cmap='viridis', alpha=0.5)
        ax[layer+1].set_title(str(layer+1))
    plt.show()