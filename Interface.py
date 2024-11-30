import streamlit as st
import numpy as np
from PIL import Image
import torch
from GradCam.GradCam import GradCAM
from AttentionMapping.RawAttention import get_rawheat_maps_from_model, return_raw_heatmap
from AttentionMapping.RollingAttention import get_rolling_heatmaps_from_model, return_rolling_heatmap
from torch import nn
from torch import optim
from torchvision.transforms import Normalize
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel

# Load the required models and preprocessors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, lr):
        super().__init__()

        class Block(nn.Module):
            def __init__(this, features):
                super().__init__()
                this.num_featues = features
                this.conv1 = nn.Conv2d(in_channels=this.num_featues, out_channels=this.num_featues, kernel_size=3, padding=1)
                this.conv2 = nn.Conv2d(in_channels=this.num_featues, out_channels=this.num_featues, kernel_size=3, padding=1)
            def forward(this, x):
                y = torch.relu(this.conv1(x))
                return torch.relu(x + y)
            
            __call__ = forward

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1), # 6 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4), # 6 x 32 x 32
            nn.Sequential(*[Block(6) for i in range(3)]),
            nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3, padding=1), # 24 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Sequential(*[Block(24) for i in range(3)])
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2) # 24 x 4 x 4

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=24*4*4, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
            nn.Sigmoid()
        )

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr, weight_decay=0.00001)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = self.linear_layers(x.reshape(-1, 24*4*4))

        return x

NORMALIZE = Normalize(
mean=(0.4883098900318146, 0.4550710618495941, 0.41694772243499756),
std =(0.25926217436790466, 0.25269168615341187, 0.25525805354118347)
)
def preprocessor(im: torch.Tensor | np.ndarray | Image.Image):
    im = Image.fromarray(np.array(im))
    im = im.convert("RGB")
    im = im.resize((128,128))
    im = NORMALIZE((torch.tensor([np.array(im)])/255.0).permute(dims=(0,3,1,2)))
    im = im.to(device)
    return im

def LoadImage(filepath:str):
    return np.array(Image.open(filepath))


# Initialize your models
CNN_MODEL = Model(lr=0.001)
CNN_MODEL.to(device)
def LoadModel(CNN_MODEL:Model):
    CNN_MODEL.load_state_dict(torch.load("GradCam/weights.pth"))
LoadModel(CNN_MODEL)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
transformer_model = ViTForImageClassification.from_pretrained('akahana/vit-base-cats-vs-dogs',attn_implementation="eager")

CAM = GradCAM(Model=CNN_MODEL, LayerName='conv_layers', Normalization=NORMALIZE)

transformer_model.eval()

# Define Streamlit app
def main():
    st.title("Attention Mapping Visualization")

    # Sidebar for selecting method
    method = st.sidebar.selectbox(
        "Select Attention Mapping Method",
        ("GradCAM", "Raw Attention", "Rolling Attention")
    )

    # Upload Image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Input for selecting layer number
        layer_number = None
        if method in ["Raw Attention", "Rolling Attention"]:
            layer_number = st.sidebar.number_input(
                "Select Layer Number",
                min_value=1,
                max_value=12,
                step=1,
                value=1,
                help="Choose the layer number to visualize the attention."
            )
        
        # Perform selected method
        if st.button("Generate HeatMap"):
            if method == "GradCAM":
                st.write("Generating GradCAM Heatmap...")
                heatmap, logits = CAM.gradCAM_heatmapAUTOLABEL_withLogits(preprocessor, image)
                st.image(heatmap, caption="GradCAM Heatmap", use_column_width=True)
                show_logits(logits)
            
            elif method == "Raw Attention":
                st.write("Generating Raw Attention Heatmap...")
                raw_map, logits = return_raw_heatmap(
                    feature_extractor=feature_extractor,
                    model=transformer_model,
                    image=image,
                    size=(224, 224),
                    layer=layer_number-1  # Pass layer number
                )
                st.image(raw_map, caption="Raw Attention Heatmap", use_column_width=True)
                show_logits(logits)
            
            elif method == "Rolling Attention":
                st.write("Generating Rolling Attention Heatmap...")
                rolling_map, logits = return_rolling_heatmap(
                    feature_extractor=feature_extractor,
                    model=transformer_model,
                    image=image,
                    size=(224, 224),
                    layers=layer_number  # Pass layer number
                )
                st.image(rolling_map, caption="Rolling Attention Heatmap", use_column_width=True)
                show_logits(logits)

def show_logits(logits):
    """Display the output label and probabilities."""
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    label = probabilities.argmax().item()
    st.write(f"Predicted Label: {label}")
    st.write("Probabilities:")
    st.write(probabilities.tolist())

# Run the Streamlit app
if __name__ == "__main__":
    main()