import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from preprocess import crop_brain_contour

class BrainTumorDataset(Dataset):
  def __init__(self, raw_data_dir):
    self.image_paths = []
    self.labels = []
    
    no_tumor_dir = os.path.join(raw_data_dir, 'no', '*.jpg')
    for path in glob.glob(no_tumor_dir):
      self.image_paths.append(path)
      self.labels.append(0)
      
    yes_tumor_dir = os.path.join(raw_data_dir, 'yes', '*.jpg')
    for path in glob.glob(yes_tumor_dir):
      self.image_paths.append(path)
      self.labels.append(1)

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    label = self.labels[idx]
    
    # 1. Preprocess:
    processed_img = crop_brain_contour(img_path, plot=False)
    
    # 2. Convert OpenCV NumPy array to PyTorch Tensor
    img_tensor = torch.tensor(processed_img, dtype=torch.float32)
    
    # 3. Reorder the dimensions
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # 4. Normalize pixels from [0, 255] down to [0.0, 1.0] for the nn
    img_tensor = img_tensor / 255.0
    
    return img_tensor, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
  dataset = BrainTumorDataset(raw_data_dir="../data/raw")
  
  # The DataLoader handles batching and shuffling the data
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  # Grab one single batch off the conveyor belt
  images, labels = next(iter(dataloader))
  
  print(f"Total images in dataset: {len(dataset)}")
  print(f"Batch Images Shape: {images.shape}") 
  print(f"Batch Labels Shape: {labels.shape}") 
  print(f"Sample Labels: {labels}")