import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset
from model import BrainTumorCNN

def train_model():
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  print(f"Training on device: {device}")

  print("Loading dataset...")
  dataset = BrainTumorDataset(raw_data_dir="../data/raw")

  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  model = BrainTumorCNN(num_classes=2).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)

  num_epochs = 10
  
  for epoch in range(num_epochs):
    model.train() # Tell the model it is learning mode (turns on Dropout)
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Grab a batch from the conveyor belt
    for batch_idx, (images, labels) in enumerate(dataloader):
      images = images.to(device)
      labels = labels.to(device)
      
      optimizer.zero_grad()
      
      # Forward Pass
      outputs = model(images)
      
      loss = criterion(outputs, labels)
      
      loss.backward()
      
      optimizer.step()
      
      # Track statistics for logging
      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total_predictions += labels.size(0)
      correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = (correct_predictions / total_predictions) * 100
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

  print("Training complete. Saving model...")
  torch.save(model.state_dict(), "brain_tumor_model.pth")
  print("Model saved successfully to src/brain_tumor_model.pth")

if __name__ == "__main__":
  train_model()