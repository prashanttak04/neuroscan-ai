import torch
import torch.nn.functional as F
from model import BrainTumorCNN
from preprocess import crop_brain_contour

def predict_tumor(image_path, model_path="brain_tumor_model.pth"):
  # 1. Load the architecture and the saved weights
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  model = BrainTumorCNN(num_classes=2)
  # weights_only=True is a security best practice when loading saved models
  model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
  model.to(device)
  
  # Set to evaluation mode! This turns off Dropout so the network uses 100% of its neurons
  model.eval() 

  # 2. Preprocess the single image
  processed_img = crop_brain_contour(image_path, plot=True)
  
  # 3. Convert OpenCV array to PyTorch Tensor
  img_tensor = torch.tensor(processed_img, dtype=torch.float32)
  img_tensor = img_tensor.permute(2, 0, 1) # Reorder to (Channels, Height, Width)
  img_tensor = img_tensor / 255.0 # Normalize
  
  # CRITICAL: The model expects a "batch", even if it's just a batch of 1.
  # unsqueeze(0) adds a fake batch dimension -> (1, 3, 227, 227)
  img_tensor = img_tensor.unsqueeze(0).to(device)

  # 4. Make the prediction
  with torch.no_grad(): # Tell PyTorch not to track gradients (saves memory)
    output = model(img_tensor)
    
    # Convert raw output into probabilities (0 to 100%)
    probabilities = F.softmax(output, dim=1)[0]
    
  # 5. Interpret the results
  no_tumor_prob = probabilities[0].item() * 100
  tumor_prob = probabilities[1].item() * 100
  
  print(f"\n--- 🧠 Diagnostic Report ---")
  print(f"File: {image_path.split('/')[-1]}")
  print(f"No Tumor: {no_tumor_prob:.2f}%")
  print(f"Tumor:    {tumor_prob:.2f}%")
  
  if tumor_prob > no_tumor_prob:
    print("=> FINAL DIAGNOSIS: TUMOR DETECTED ⚠️")
  else:
    print("=> FINAL DIAGNOSIS: HEALTHY BRAIN ✅")

if __name__ == "__main__":
  # Let's test it! Point this to any image in your dataset.
  # Try testing one from 'yes' and one from 'no' to see how confident it is.
  test_image = "../data/raw/test2.jpg"
  predict_tumor(test_image)