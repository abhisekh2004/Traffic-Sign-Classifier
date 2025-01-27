import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import torch.nn.functional as F

# Define the class mapping for traffic signs
class_map = {
    0: "Speed limit (20km/h)", 
    1: "Speed limit (30km/h)", 
    12: "Speed limit (50km/h)", 
    23: "Speed limit (60km/h)", 
    34: "Speed limit (70km/h)", 
    38: "Speed limit (80km/h)", 
    39: "End of speed limit (80km/h)", 
    40: "Speed limit (100km/h)", 
    41: "Speed limit (120km/h)", 
    42: "No passing", 
    2: "No passing for vehicles over 3.5 metric tons", 
    3: "Right-of-way at the next intersection", 
    4: "Priority road", 
    5: "Yield", 
    6: "Stop", 
    7: "No vehicles", 
    8: "Vehicles over 3.5 metric tons prohibited", 
    9: "No entry", 
    10: "General caution", 
    11: "Dangerous curve to the left", 
    13: "Dangerous curve to the right", 
    14: "Double curve", 
    15: "Bumpy road", 
    16: "Slippery road", 
    17: "Road narrows on the right", 
    18: "Road work", 
    19: "Traffic signals", 
    20: "Pedestrians", 
    21: "Children crossing", 
    22: "Bicycles crossing", 
    24: "Beware of ice/snow", 
    25: "Wild animals crossing", 
    26: "End of all speed and passing limits", 
    27: "Turn right ahead", 
    28: "Turn left ahead", 
    29: "Ahead only", 
    30: "Go straight or right", 
    31: "Go straight or left", 
    32: "Keep right", 
    33: "Keep left", 
    35: "Roundabout mandatory", 
    36: "End of no passing", 
    37: "End of no passing by vehicles over 3.5 metric tons"
}

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define the custom classifier
class Dc_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 43)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Load the model
model_path = "Models/resnet_custom_fc_3.pth"
model = models.resnet18(pretrained=False)
model.fc = Dc_model()  # Replace the final layer
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Traffic Sign Classifier")
st.write("Upload an image of a traffic sign to classify it.")

# File uploader
uploaded_file = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_tensor = preprocess_image(image).to("cpu")

    # Predict the class
    with torch.no_grad():
        raw_output = model(image_tensor)

    # Get the predicted class and confidence
    predicted_index = torch.argmax(raw_output, dim=1).item()
    confidence = F.softmax(raw_output, dim=1)[0][predicted_index].item()
    class_name = class_map[predicted_index]

    # Display the results
    st.write(f"**Predicted Class**: {class_name}")
    st.write(f"**Confidence**: {confidence*100:.2f}%")
