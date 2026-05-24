import torch
import torch.nn.functional as F
from model import BrainTumorCNN
from preprocess import crop_brain_contour

def predict_tumor(image_path, model_path="brain_tumor_model.pth"):
  # Load the architecture and the saved weights
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  model = BrainTumorCNN(num_classes=2)

  model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
  model.to(device)
  
  model.eval() 

  processed_img = crop_brain_contour(image_path, plot=True)
  
  # Convert OpenCV array to PyTorch Tensor
  img_tensor = torch.tensor(processed_img, dtype=torch.float32)
  img_tensor = img_tensor.permute(2, 0, 1)
  img_tensor = img_tensor / 255.0
  
  img_tensor = img_tensor.unsqueeze(0).to(device)

  with torch.no_grad():
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)[0]
    
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
  test_image = "../data/raw/test2.jpg"
  predict_tumor(test_image)