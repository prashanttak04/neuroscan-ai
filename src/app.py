import streamlit as st
import torch
import torch.nn.functional as F
import os
from model import BrainTumorCNN
from preprocess import crop_brain_contour

st.set_page_config(page_title="NeuroScan AI", layout="wide")

st.markdown("""
  <style>
  /* Importing a premium Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* Force Streamlit to use it everywhere */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  
  .stApp {
    background-color: #F4F7F6;
  }
  
  [data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 600;
    color: #0D9488 !important;
  }
  
  [data-testid="stMetricLabel"] {
    font-size: 14px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .stImage > img {
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  }
  
  .stAlert {
    border-radius: 8px;
    border-left: 5px solid #0D9488;
    background-color: #FFFFFF;
    color: #1E293B;
  }

  /* --- NEW: Sidebar Card Button CSS --- */
  [data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    color: #1E293B;
    border-radius: 12px;
    padding: 24px 20px; /* Makes the button taller like a card */
    margin-bottom: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    font-size: 16px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
  }
  
  [data-testid="stSidebar"] .stButton > button:hover {
    border-color: #0D9488;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.15);
    color: #0D9488;
  }
  </style>
  """, unsafe_allow_html=True)

if "current_page" not in st.session_state:
  st.session_state.current_page = "portal"

with st.sidebar:
  st.markdown("### Resource Hub")
  
  if st.button("Diagnostic Portal", use_container_width=True):
    st.session_state.current_page = "portal"
    
  if st.button("User Documentation", use_container_width=True):
    st.session_state.current_page = "guide"
  
  st.divider()
  
  st.markdown("### System Telemetry")
  st.markdown("**Status:** Operational")
  st.markdown("**Mode:** Inference")
  st.markdown("**Architecture:** Custom 22-Layer CNN")
  st.divider()
  st.markdown("### Hardware")
  device_name = "Apple Silicon GPU (MPS)" if torch.backends.mps.is_available() else "CPU"
  st.info(f"Computing on: {device_name}")


if st.session_state.current_page == "guide":
  # ==========================================
  # INSTRUCTION MANUAL PAGE
  # ==========================================
  st.markdown("<h1 style='color: #0F172A; margin-bottom: 0;'>User Documentation</h1>", unsafe_allow_html=True)
  st.markdown("<p style='color: #64748B; font-size: 16px; margin-top: 0;'>Clinical Image Inference Portal | NeuroScan AI</p>", unsafe_allow_html=True)
  st.divider()
  
  st.markdown("### 1. Overview")
  st.write("""
    NeuroScan AI is a deep learning diagnostic tool built to analyze Magnetic Resonance Imaging (MRI) scans of the human brain. 
    It utilizes a custom 22-layer Convolutional Neural Network (CNN) trained to detect structural anomalies that may indicate the presence of a tumor.
  """)
  
  st.markdown("### 2. Step-by-Step Guide")
  st.write("**Step 1: Obtain an MRI Scan**")
  st.write("Ensure you have a top-down (axial) MRI scan saved as a standard digital image format (`.jpg`, `.jpeg`, or `.png`).")
  
  st.write("**Step 2: Upload to the Portal**")
  st.write("Navigate to the **Diagnostic Portal** using the sidebar on the left. Drag and drop your image into the upload box.")
  
  st.write("**Step 3: Automated Preprocessing**")
  st.write("Once uploaded, our computer vision pipeline automatically strips away background noise, crops the skull contour, and standardizes the pixel matrix for the neural network.")
  
  st.write("**Step 4: Inference & Review**")
  st.write("The AI will output two probability scores. A score above 50% in the 'Tumor Detected' category will trigger a high-risk alert.")
  
  st.divider()
  st.warning("**MEDICAL DISCLAIMER:** This software is a developmental Machine Learning prototype. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified radiologist or physician with any questions you may have regarding a medical condition.")


elif st.session_state.current_page == "portal":
  # ==========================================
  # MAIN INFERENCE PORTAL PAGE
  # ==========================================
  st.markdown("<h1 style='color: #0F172A; margin-bottom: 0;'>NEUROSCAN AI</h1>", unsafe_allow_html=True)
  st.markdown("<p style='color: #64748B; font-size: 16px; margin-top: 0;'>Clinical Image Inference Portal | Model: BrainTumorCNN v1.0</p>", unsafe_allow_html=True)
  st.divider()

  left_column, right_column = st.columns([1, 1.2], gap="large")

  with left_column:
    st.markdown("### Upload MRI Scan")
    uploaded_file = st.file_uploader("Select a JPEG or PNG image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
      st.image(uploaded_file, caption="Input Scan", use_container_width=True)
      
      temp_path = "temp_processing_node.jpg"
      with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

  with right_column:
    st.markdown("### Diagnostic Analysis")
    
    if uploaded_file is not None:
      @st.cache_resource
      def load_frozen_weights():
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        net = BrainTumorCNN(num_classes=2)
        
        # --- NEW DYNAMIC PATH LOGIC ---
        # Get the absolute directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "brain_tumor_model.pth")
        
        net.load_state_dict(torch.load(model_path, map_location=dev, weights_only=True))
        net.to(dev)
        net.eval()
        return net, dev

      try:
        with st.spinner("Preprocessing image..."):
          model, device = load_frozen_weights()
          
          processed_img = crop_brain_contour(temp_path, plot=False)
          
          img_tensor = torch.tensor(processed_img, dtype=torch.float32)
          img_tensor = img_tensor.permute(2, 0, 1) / 255.0
          img_tensor = img_tensor.unsqueeze(0).to(device)
          
        with st.spinner("Running model inference..."):
          with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            
        no_tumor_prob = probabilities[0].item() * 100
        tumor_prob = probabilities[1].item() * 100
        
        stat_col1, stat_col2 = st.columns(2)
        stat_col1.metric("No Tumor (Probability)", f"{no_tumor_prob:.2f}%")
        stat_col2.metric("Tumor Detected (Probability)", f"{tumor_prob:.2f}%")
        
        st.divider()
        
        st.markdown("### Diagnostic Report")
        if tumor_prob > no_tumor_prob:
          st.error("**TUMOR DETECTED:** High probability anomaly detected. Please consult a radiologist.")
        else:
          st.success("**NO TUMOR DETECTED:** Scan is within normal parameters.")
          
      except Exception as error_log:
        st.error(f"Error during processing: {error_log}")
        
      finally:
        if os.path.exists(temp_path):
          os.remove(temp_path)
          
    else:
      st.info("Awaiting MRI scan upload.")