import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image

# --- CRASH FIX FOR WINDOWS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Redefine your exact model architecture framework so PyTorch can load the saved weights
class ImageClassifierCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ImageClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 18 * 18, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 32 * 18 * 18)
        x = self.fc(x)
        return x

# 2. Cache the model loading routine so the web page stays lightning fast
@st.cache_resource
def load_trained_model():
    model = ImageClassifierCNN(num_classes=4)
    # Load the best weights we saved during training onto the CPU
    model.load_state_dict(torch.load('best_image_classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_trained_model()
class_names = ['cars', 'cats', 'dogs', 'flowers']

# 3. Streamlit UI Layout Configuration
st.set_page_config(page_title="CNN Image Classifier", page_icon="🔮", layout="centered")

st.title("🔮 CNN Image Classification System")
st.markdown("""
    Upload an image belonging to one of the target classes (**Cars, Cats, Dogs, Flowers**). 
    The custom-trained deep learning model will analyze the structural patterns and output its prediction.
""")
st.write("---")

# File uploader drag-and-drop widget component
uploaded_file = st.file_uploader("Drop your image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the user's uploaded image on the web dashboard
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image Preview", use_container_width=True)
    
    st.write("")
    # Add a clickable button to trigger the calculation
    if st.button("🚀 Run Neural Network Inference"):
        
        # Match data-processing dimensions exactly to our original settings
        inference_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Preprocess and append the mini-batch dimension via unsqueeze
        img_tensor = inference_transform(image).float().unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # Apply softmax to translate raw output logits into clean confidence percentages
            probabilities = F.softmax(outputs, dim=1)
            confidence, index = torch.max(probabilities, 1)
            
        predicted_class = class_names[index.item()]
        confidence_percentage = confidence.item() * 100
        
        # Display polished visual results cards
        st.success(f"### Prediction Result: **{predicted_class.upper()}**")
        st.metric(label="Model Prediction Confidence Score", value=f"{confidence_percentage:.2f}%")