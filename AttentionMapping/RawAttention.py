from typing import Iterable
from torchvision.transforms import Resize
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def raw_attention_map(attentions:Iterable[torch.Tensor], layer:int = -1, size:tuple[int] = (224,224)) -> np.ndarray:
    avg_attention = attentions[layer].mean(dim=1).squeeze(0).detach().numpy()
    # We need attention from Cls token to all other tokens, which is obtained from the first row of the attention matrix. So here we will use the first layer
    # of the avg_attention matrix. Also since we only need attentions from cls token to patch tokens, we will also ignore the first element of the first row which is
    # just attention from class token to class token
    cls_attention = avg_attention[0, 1:]
    cls_attention = cls_attention.reshape(int(np.sqrt(cls_attention.shape[0])), int(np.sqrt(cls_attention.shape[0])))
    resize = Resize(size)
    cls_attention = resize(torch.tensor(cls_attention).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())
    return cls_attention

# Image.Image is PIL.Image.Image, just incase if you face import errors @dhruv
def raw_attention_heatmap(attentions:Iterable[torch.Tensor], image: Image.Image, layer:int = -1, size:tuple[int] = (224,224)) -> None:
    attention_map = raw_attention_map(attentions, layer, size)
    resize = Resize(size)
    image = resize(image)
    plt.imshow(image)
    plt.imshow(attention_map, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Attention Intensity')
    plt.show()

def get_rawheat_maps_from_model(feature_extractor, model, image: Image.Image, layer:int = -1, size:tuple[int] = (224, 224)) -> None:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True, return_dict=True)
    attentions = outputs.attentions
    print("output class probabilities:")
    print(nn.functional.softmax(outputs.logits[0], dim=0).detach().numpy())
    raw_attention_heatmap(attentions, image, layer, size)

def raw_heatmaps_all_layer(feature_extractor, model, image: Image.Image, size:tuple[int] = (224, 224)) -> None:
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
        attention_map = raw_attention_map(attentions, layer, size)
        ax[layer+1].imshow(image)
        ax[layer+1].imshow(attention_map, cmap='viridis', alpha=0.5)
        ax[layer+1].set_title(str(layer+1))
    plt.show()