# app/streamlit_app.py
import streamlit as st
import torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

MODELS = Path("../models")

# ----- models (same as notebook) -----
class TabularMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x)

def get_img_encoder():
    base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in base.parameters(): p.requires_grad = False
    in_feat = base.fc.in_features
    base.fc = nn.Identity()
    return base, in_feat

class FusedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, feat_dim = get_img_encoder()
        self.tab_mlp = TabularMLP()
        self.fc = nn.Linear(feat_dim + 1, 1)
    def forward(self, img, tab):
        img_feat = self.backbone(img)
        tab_feat = self.tab_mlp(tab)
        return self.fc(torch.cat([img_feat, tab_feat], dim=1))

@st.cache_resource
def load_model():
    m = FusedModel()
    m.load_state_dict(torch.load(MODELS/"fused.pth", map_location="cpu"))
    m.eval()
    return m

# ----- transforms & CAM -----
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def gradcam(model, img_t, target_layer):
    acts={}
    def fwd(_, __, out):
        acts["v"]=out
    h = target_layer.register_forward_hook(fwd)
    img_t.requires_grad_(True)
    out = model(img_t, torch.zeros((1,3)))
    logit = out.mean()
    grads = torch.autograd.grad(logit, acts["v"])[0]
    h.remove()
    A, G = acts["v"].detach(), grads.detach()
    w = G.mean(dim=(2,3), keepdim=True)
    cam = torch.relu((w*A).sum(dim=1)).squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam

def overlay(img_rgb, cam):
    cam = cv2.resize(cam, (img_rgb.size[0], img_rgb.size[1]))
    heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
    mix = (0.4*heat + 0.6*np.asarray(img_rgb)).clip(0,255).astype(np.uint8)
    return Image.fromarray(mix)

# ----- UI -----
st.title("SeleneX — Ovarian Ultrasound + Biomarker Demo")
model = load_model()

img_file = st.file_uploader("Upload ultrasound image", type=["png","jpg","jpeg"])
age = st.slider("Age", 28, 78, 52)
ca125 = st.number_input("CA-125 (U/mL)", min_value=0.0, max_value=1000.0, value=35.0, step=1.0)
brca = st.selectbox("BRCA", [0,1], index=0)

if img_file:
    img = Image.open(img_file).convert("RGB").resize((224,224))
    x_img = transform(img).unsqueeze(0)
    x_tab = torch.tensor([[age, ca125, brca]], dtype=torch.float32)

    with torch.no_grad():
        logit = model(x_img, x_tab)
        prob = torch.sigmoid(logit).item()

    st.metric("Malignancy Probability", f"{prob:.2f}")

    # Grad-CAM
    cam = gradcam(model, x_img.clone(), model.backbone.layer4[-1])
    st.image(overlay(img, cam), caption="Grad-CAM")
