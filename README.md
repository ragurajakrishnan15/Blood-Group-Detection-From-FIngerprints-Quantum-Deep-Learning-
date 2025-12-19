# Blood Group Detection From Fingerprints - Quantum Deep Learning

A cutting-edge deep learning application that predicts blood groups from fingerprint images using hybrid quantum-classical neural networks.

## 🩸 Overview

This project leverages the power of quantum computing and deep learning to classify blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-) from fingerprint images. The application combines traditional Convolutional Neural Networks (CNN) with Quantum Neural Networks (QNN) for enhanced performance.

## ✨ Features

- **Hybrid Quantum-Classical Model**: Combines CNN with PennyLane quantum circuits
- **Simple CNN Model**: Traditional deep learning approach for comparison
- **Interactive Web Interface**: Built with Streamlit for easy image upload and prediction
- **Real-time Predictions**: Instant blood group classification with confidence scores
- **Visual Analytics**: Interactive visualizations using Plotly

## 🔧 Technologies Used

- **Python 3.13**
- **PyTorch**: Deep learning framework
- **PennyLane**: Quantum machine learning
- **Streamlit**: Web interface
- **Pillow**: Image processing
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ragurajakrishnan15/Blood-Group-Detection-From-FIngerprints-Quantum-Deep-Learning-.git
   cd Blood-Group-Detection-From-FIngerprints-Quantum-Deep-Learning-
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Files**
   
   The trained model files are not included in the repository due to their large size (590MB total). You need to:
   - Train the models using the Jupyter notebook (`ai-project-final (1).ipynb`), OR
   - Download pre-trained models from [releases/cloud storage] and place them in the project directory:
     - `simple_cnn_blood_group.pkl`
     - `hybrid_cnn_blood_group.pkl`

## 🚀 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Jupyter Notebook

Open and run the notebook to:
- Train the models from scratch
- Visualize training metrics
- Experiment with different architectures

```bash
jupyter notebook "ai-project-final (1).ipynb"
```

## 📊 Model Architecture

### Simple CNN
- Convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Fully connected layers for classification

### Hybrid Quantum-CNN
- CNN feature extraction layers
- Quantum circuit layer using PennyLane
- Classical output layer for final classification

## 🎯 Blood Group Classes

The model classifies fingerprints into 8 blood group categories:
- A+ (A Positive)
- A- (A Negative)
- B+ (B Positive)
- B- (B Negative)
- AB+ (AB Positive)
- AB- (AB Negative)
- O+ (O Positive)
- O- (O Negative)

## 📁 Project Structure

```
Blood-Group-Detection-From-FIngerprints-Quantum-Deep-Learning-/
│
├── app.py                          # Streamlit web application
├── ai-project-final (1).ipynb      # Training notebook
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
├── simple_cnn_blood_group.pkl      # Trained Simple CNN model (not in repo)
└── hybrid_cnn_blood_group.pkl      # Trained Hybrid model (not in repo)
```

## 🔬 Training the Models

To train the models from scratch:

1. Open the Jupyter notebook
2. Prepare your fingerprint dataset organized by blood group
3. Run all cells in sequence
4. Models will be saved as `.pkl` files

## 🖼️ Using the Web Interface

1. Launch the app with `streamlit run app.py`
2. Select a model (Simple CNN or Hybrid Quantum-CNN)
3. Upload a fingerprint image
4. View the predicted blood group with confidence scores
5. See visualization of prediction probabilities

## 📝 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- Minimum 8GB RAM
- 2GB disk space for models

## ⚠️ Important Notes

- Model accuracy depends on the quality and diversity of training data
- Fingerprint images should be clear and well-lit
- This is a research/educational project and should not be used for medical diagnosis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Raguraja Krishnan**
- GitHub: [@ragurajakrishnan15](https://github.com/ragurajakrishnan15)

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- PennyLane for quantum machine learning framework
- PyTorch community for deep learning tools
- Streamlit for the amazing web framework

---

**Note**: This is a research project. Always consult medical professionals for actual blood group determination.
