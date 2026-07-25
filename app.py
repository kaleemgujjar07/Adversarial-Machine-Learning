import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# 1. Load the Model and Classes
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    
    # Download the model from Hugging Face Hub
    # REPLACE 'YOUR_HF_USERNAME' with your actual Hugging Face username!
    model_path = hf_hub_download(repo_id='gujjarkaleem37/cifar-resnet18', filename='cifar_resnet18.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()
fmodel = fb.PyTorchModel(model, bounds=(-1, 1))

# Classes are hardcoded here, so we don't need classes.json
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Streamlit UI
st.title("🧠 Adversarial Machine Learning Demo (CIFAR-10)")
st.markdown("This app demonstrates how an FGSM attack can trick a Deep Learning model into making wrong predictions by adding invisible noise to an image.")
st.write("---")

# 3. Load CIFAR-10 Test Set
@st.cache_resource
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return testset

testset = load_data()

# 4. User Controls
col1, col2 = st.columns(2)
with col1:
    img_index = st.slider("Select an Image Index", 0, 9999, 0)
with col2:
    epsilon = st.slider("Attack Strength (Epsilon)", 0.0, 0.1, 0.03, 0.01)

# Get the selected image and label
test_image, test_label = testset[img_index]
test_image = test_image.unsqueeze(0) # Add batch dimension
test_label = torch.tensor([test_label])

# 5. Generate Predictions
# Before Attack
outputs = fmodel(test_image)
original_pred = outputs.argmax(axis=1).item()
original_conf = torch.nn.functional.softmax(outputs, dim=1)[0, original_pred].item()

# Generate Attack
attack = fb.attacks.FGSM()
_, adv_image, _ = attack(fmodel, test_image, test_label, epsilons=epsilon)

# After Attack
outputs_adv = fmodel(adv_image)
adv_pred = outputs_adv.argmax(axis=1).item()
adv_conf = torch.nn.functional.softmax(outputs_adv, dim=1)[0, adv_pred].item()

# 6. Visualize Everything
def tensor_to_img(tensor):
    img = tensor.squeeze().detach().numpy()
    img = img / 2 + 0.5  # unnormalize
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)

# Calculate the noise (perturbation)
perturbation = adv_image - test_image

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Original Image
ax[0].imshow(tensor_to_img(test_image))
ax[0].set_title(f"Original Image\nTrue: {classes[test_label.item()]}\nPred: {classes[original_pred]} ({original_conf*100:.1f}%)", color="green")
ax[0].axis('off')

# Perturbation (The Noise)
perturbation_vis = perturbation.squeeze().detach().numpy()
perturbation_vis = perturbation_vis / (2 * epsilon + 1e-8) + 0.5
ax[1].imshow(np.clip(np.transpose(perturbation_vis, (1, 2, 0)), 0, 1))
ax[1].set_title("The Added Noise\n(Invisible to humans)")
ax[1].axis('off')

# Adversarial Image
ax[2].imshow(tensor_to_img(adv_image))
if original_pred != adv_pred:
    ax[2].set_title(f"Hacked Image\nTrue: {classes[test_label.item()]}\nPred: {classes[adv_pred]} ({adv_conf*100:.1f}%)", color="red")
else:
    ax[2].set_title(f"Hacked Image\nPred: {classes[adv_pred]} ({adv_conf*100:.1f}%)\n(Attack failed, increase epsilon)", color="orange")
ax[2].axis('off')

st.pyplot(fig)

st.write("---")
if original_pred != adv_pred:
    st.error(f"🚨 ATTACK SUCCESSFUL! The AI changed its prediction from '{classes[original_pred]}' to '{classes[adv_pred]}'.")
else:
    st.warning("The attack did not change the prediction. Try increasing the Attack Strength (Epsilon) or selecting a different image.")
