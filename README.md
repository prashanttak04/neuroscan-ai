# NeuroScan AI: Custom CNN for Brain Tumor Detection 🧠

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io/)
[![Deployment](https://img.shields.io/badge/Deployed-Streamlit_Cloud-success)](https://streamlit.io/cloud)

**Live Application:** [View the Live Diagnostic Portal Here](https://neuroscan-ai-scratch-cnn.streamlit.app/)

## Overview
NeuroScan AI is an end-to-end deep learning diagnostic tool built to analyze Magnetic Resonance Imaging (MRI) scans. It utilizes a **custom-built 22-layer Convolutional Neural Network (CNN) designed from scratch** to detect structural anomalies indicating the presence of a brain tumor. 

The inference model is wrapped in a high-performance, responsive Streamlit dashboard configured with a custom clinical UI/UX.

## Core Features
* **Custom Architecture:** Built entirely from scratch using PyTorch (no pre-trained weights used for V1 binary classification).
* **Automated Preprocessing:** Integrates OpenCV to automatically strip background noise and crop skull contours before tensor normalization.
* **Clinical UI/UX:** Styled via custom CSS injection to mimic a professional radiology dashboard with full telemetry and diagnostic reporting.
* **Edge-Ready Inference:** Optimized to run strict CPU inference for cloud deployment scalability.

## Tech Stack
* **Deep Learning Framework:** PyTorch, TorchVision
* **Computer Vision:** OpenCV (`opencv-python-headless`)
* **Scientific Computing:** NumPy
* **Frontend / Deployment:** Streamlit, Streamlit Community Cloud

## Local Installation
To run this application on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/prashanttak04/neuroscan-ai.git](https://github.com/prashanttak04/neuroscan-ai.git)
   cd neuroscan-ai

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. Launch the dashboard:
   ```bash
   cd src
   streamlit run app.py

## Disclaimer
This software is a developmental Machine Learning prototype. It is not a substitute for professional medical advice, diagnosis, or treatment.
