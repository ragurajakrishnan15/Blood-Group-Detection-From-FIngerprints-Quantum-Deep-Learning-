import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import plotly.express as px
import pandas as pd
import pennylane as qml
from torchvision import transforms
import logging
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
qml.numpy.random.seed(seed)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model file paths
MODEL_DIR = r"C:\Users\ragu\Desktop\blood gr\blood gr"
SIMPLE_CNN_PATH = os.path.join(MODEL_DIR, "simple_cnn_blood_group.pkl")
HYBRID_CNN_PATH = os.path.join(MODEL_DIR, "hybrid_cnn_blood_group.pkl")

# Class names (corrected order to match Notebook)
class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Custom transform for contrast enhancement
class AdjustContrast:
    def __init__(self, factor=2.0):
        self.factor = factor
    def __call__(self, img):
        return transforms.functional.adjust_contrast(img, self.factor)

# Test transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    AdjustContrast(2.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 8)
        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        shortcut = self.shortcut(x2)
        x3 = F.relu(self.bn3(self.conv3(x2)) + shortcut)
        x4 = self.pool(F.relu(self.bn4(self.conv4(x3))))
        x5 = x4.view(-1, 128 * 28 * 28)
        x6 = F.relu(self.fc1(x5))
        x7 = self.dropout(x6)
        x8 = self.fc2(x7)
        return x8

# Define quantum circuit and HybridCNN
n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (2, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)

class HybridCNN(nn.Module):
    def __init__(self):
        super(HybridCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc_quantum = nn.Linear(64, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.3)
        self.q_layer = QuantumLayer()
        self.fc3 = nn.Linear(4, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        shortcut = self.shortcut(x)
        x = F.relu(self.bn3(self.conv3(x)) + shortcut)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_quantum(x))
        x = self.bn5(x)
        x = self.q_layer(x)
        x = self.fc3(x)
        return x

# Test single image function
def test_single_image(image, model, model_name, transform, class_names):
    try:
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]
            prob_list = {class_names[i]: probs[0][i].item() for i in range(len(class_names))}
        return predicted_class, prob_list
    except Exception as e:
        logger.error(f"Error processing image with {model_name}: {e}")
        return None, None

# Initialize models
model_classical = SimpleCNN().to(device)
model_hybrid = HybridCNN().to(device)

# Load models
try:
    model_classical.load_state_dict(torch.load(SIMPLE_CNN_PATH, map_location=device))
except Exception as e:
    st.error(f"Failed to load SimpleCNN model: {str(e)}")
    st.stop()

try:
    model_hybrid.load_state_dict(torch.load(HYBRID_CNN_PATH, map_location=device))
except Exception as e:
    st.error(f"Failed to load HybridCNN model: {str(e)}")
    st.stop()

# Custom CSS for Apple-like premium look
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&display=swap');

body {
    background-color: #000;
    color: #ffffff;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 0;
}
.stApp {
    background: linear-gradient(135deg, #1c2526 0%, #0a0e0f 100%);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
    max-width: 1200px;
    margin: 0 auto;
}
h1 {
    color: #f5f5f7;
    font-size: 3em;
    font-weight: 600;
    text-align: center;
    margin-bottom: 40px;
    letter-spacing: -0.02em;
}
.stFileUploader {
    background-color: #2c2c2e;
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.stButton > button {
    background-color: #ffd700;
    color: #000;
    border: none;
    border-radius: 30px;
    padding: 12px 30px;
    font-size: 1.1em;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
}
.stButton > button:hover {
    background-color: #ffea00;
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(255, 215, 0, 0.5);
}
.stSpinner > div {
    color: #ffd700;
}
.uploaded-image-container {
    display: flex;
    justify-content: center;
    margin: 30px 0;
}
.uploaded-image {
    max-width: 350px;
    border-radius: 15px;
    border: 2px solid rgba(255, 215, 0, 0.2);
    transition: transform 0.3s ease;
}
.uploaded-image:hover {
    transform: scale(1.03);
}
.image-caption {
    color: #d1d1d6;
    font-size: 0.9em;
    text-align: center;
    margin-top: 10px;
}
.results-container {
    display: flex;
    justify-content: space-between;
    gap: 30px;
    margin-top: 40px;
    flex-wrap: wrap;
}
.model-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    flex: 1;
    min-width: 400px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.model-card h3 {
    color: #ffd700;
    font-size: 1.5em;
    font-weight: 500;
    margin-bottom: 15px;
}
.predicted-class {
    font-size: 1.3em;
    color: #f5f5f7;
    margin-bottom: 20px;
}
.predicted-class span {
    color: #ffd700;
    font-weight: 600;
}
.probability-table th, .probability-table td {
    color: #d1d1d6 !important;
    border-color: rgba(255, 215, 0, 0.2) !important;
    font-size: 0.9em;
}
@media (max-width: 768px) {
    .results-container {
        flex-direction: column;
        gap: 20px;
    }
    .model-card {
        min-width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Blood Group Predictor")

# Image upload
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

if uploaded_file is not None:
    # Convert image to base64 for custom HTML rendering
    image = Image.open(uploaded_file).convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Display image with custom HTML/CSS
    st.markdown(
        f"""
        <div class="uploaded-image-container">
            <img src="data:image/png;base64,{img_str}" class="uploaded-image" alt="Uploaded Image">
        </div>
        <p class="image-caption">Uploaded Image</p>
        """,
        unsafe_allow_html=True
    )

    # Predict button
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            # Run predictions for both models
            predicted_class_simple, prob_list_simple = test_single_image(image, model_classical, "SimpleCNN", test_transform, class_names)
            predicted_class_hybrid, prob_list_hybrid = test_single_image(image, model_hybrid, "HybridCNN", test_transform, class_names)
            
            if predicted_class_simple is None or predicted_class_hybrid is None:
                st.error("Prediction failed for one or both models. Please try again.")
            else:
                # Display results in two side-by-side cards
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                
                # SimpleCNN Results
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown('<h3>SimpleCNN Prediction</h3>', unsafe_allow_html=True)
                st.markdown(f'<p class="predicted-class">Predicted Blood Group: <span>{predicted_class_simple}</span></p>', unsafe_allow_html=True)
                
                # SimpleCNN Probabilities Table
                st.markdown('<h4 style="color: #ffd700; font-size: 1.2em; margin-top: 20px;">Probabilities</h4>', unsafe_allow_html=True)
                prob_df_simple = pd.DataFrame({
                    "Blood Group": list(prob_list_simple.keys()),
                    "Probability (%)": [prob * 100 for prob in prob_list_simple.values()]
                })
                st.table(prob_df_simple.style.set_table_attributes('class="probability-table"'))
                
                # SimpleCNN Probabilities Plot
                fig_simple = px.bar(
                    prob_df_simple,
                    x="Probability (%)",
                    y="Blood Group",
                    orientation='h',
                    title="",
                    color="Probability (%)",
                    color_continuous_scale="YlOrBr",
                    range_x=[0, 100],
                    height=300
                )
                fig_simple.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#d1d1d6",
                    xaxis_title="Probability (%)",
                    yaxis_title="Blood Group",
                    margin=dict(l=40, r=40, t=20, b=20),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
                )
                fig_simple.update_traces(
                    hovertemplate="%{y}: %{x:.2f}%",
                    marker_line_color="#ffd700",
                    marker_line_width=1.5
                )
                st.plotly_chart(fig_simple, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # HybridCNN Results
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown('<h3>HybridCNN Prediction</h3>', unsafe_allow_html=True)
                st.markdown(f'<p class="predicted-class">Predicted Blood Group: <span>{predicted_class_hybrid}</span></p>', unsafe_allow_html=True)
                
                # HybridCNN Probabilities Table
                st.markdown('<h4 style="color: #ffd700; font-size: 1.2em; margin-top: 20px;">Probabilities</h4>', unsafe_allow_html=True)
                prob_df_hybrid = pd.DataFrame({
                    "Blood Group": list(prob_list_hybrid.keys()),
                    "Probability (%)": [prob * 100 for prob in prob_list_hybrid.values()]
                })
                st.table(prob_df_hybrid.style.set_table_attributes('class="probability-table"'))
                
                # HybridCNN Probabilities Plot
                fig_hybrid = px.bar(
                    prob_df_hybrid,
                    x="Probability (%)",
                    y="Blood Group",
                    orientation='h',
                    title="",
                    color="Probability (%)",
                    color_continuous_scale="YlOrBr",
                    range_x=[0, 100],
                    height=300
                )
                fig_hybrid.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#d1d1d6",
                    xaxis_title="Probability (%)",
                    yaxis_title="Blood Group",
                    margin=dict(l=40, r=40, t=20, b=20),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
                )
                fig_hybrid.update_traces(
                    hovertemplate="%{y}: %{x:.2f}%",
                    marker_line_color="#ffd700",
                    marker_line_width=1.5
                )
                st.plotly_chart(fig_hybrid, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)