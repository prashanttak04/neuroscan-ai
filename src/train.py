import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom files!
from dataset import BrainTumorDataset
from model import BrainTumorCNN

def train_model():
  # 1. Hardware setup: Use Apple Silicon GPU (MPS) if available, otherwise CPU
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  print(f"Training on device: {device}")

  # 2. Set up the Conveyor Belt (DataLoader)
  print("Loading dataset...")
  dataset = BrainTumorDataset(raw_data_dir="../data/raw")
  # We use batch_size=32 to feed 32 images at a time
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  # 3. Set up the Brain (Model) and move it to the GPU
  model = BrainTumorCNN(num_classes=2).to(device)

  # 4. Set up the Judge (Loss) and the Mechanic (Optimizer)
  criterion = nn.CrossEntropyLoss()
  # Learning rate (lr) determines how drastically the mechanic changes the weights
  optimizer = optim.Adam(model.parameters(), lr=0.0001)

  # 5. The Intermediator Loop (Training for 10 Epochs)
  num_epochs = 10
  
  for epoch in range(num_epochs):
    model.train() # Tell the model it is learning mode (turns on Dropout)
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Grab a batch from the conveyor belt
    for batch_idx, (images, labels) in enumerate(dataloader):
      # Move the images and labels to the Mac GPU
      images = images.to(device)
      labels = labels.to(device)

      # --- THE 5 HOLY STEPS OF PYTORCH TRAINING ---
      
      # Step A: Clear the mechanic's tools from the last batch
      optimizer.zero_grad()
      
      # Step B: Forward Pass (Brain makes a guess)
      outputs = model(images)
      
      # Step C: Calculate the Error (Judge scores the guess)
      loss = criterion(outputs, labels)
      
      # Step D: Backward Pass (Calculate how to fix the weights)
      loss.backward()
      
      # Step E: Update Weights (Mechanic applies the fixes)
      optimizer.step()
      
      # --------------------------------------------

      # Track statistics for logging
      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total_predictions += labels.size(0)
      correct_predictions += (predicted == labels).sum().item()

    # Calculate metrics at the end of the epoch
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = (correct_predictions / total_predictions) * 100
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

  # Save the trained brain to a file so we can use it later!
  print("Training complete. Saving model...")
  torch.save(model.state_dict(), "brain_tumor_model.pth")
  print("Model saved successfully to src/brain_tumor_model.pth")

if __name__ == "__main__":
  train_model()