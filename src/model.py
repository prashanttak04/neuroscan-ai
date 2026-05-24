import torch
import torch.nn as nn

class BrainTumorCNN(nn.Module):
  def __init__(self, num_classes=2):
    super(BrainTumorCNN, self).__init__()
    
    # Block 1 (Layers 1-5): Initial Feature Extraction
    self.block1 = nn.Sequential(
      # Layer 1-3: Convolution (128 filters, 6x6 kernel) + ReLU
      nn.Conv2d(in_channels=3, out_channels=128, kernel_size=6, stride=4, padding=0),
      nn.ReLU(inplace=True),
      # Layer 4: Cross Channel Normalization (Local Response Norm)
      nn.LocalResponseNorm(size=5),
      # Layer 5: Max Pooling (2x2)
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    # Block 2 (Layers 6-8)
    self.block2 = nn.Sequential(
      # Layer 6-7: Convolution (96 filters, 6x6 kernel) + ReLU
      nn.Conv2d(in_channels=128, out_channels=96, kernel_size=6, stride=1, padding=2),
      nn.ReLU(inplace=True),
      # Layer 8: Max Pooling
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    # Block 3 (Layers 9-11)
    self.block3 = nn.Sequential(
      # Layer 9-10: Convolution (96 filters, 2x2 kernel) + ReLU
      nn.Conv2d(in_channels=96, out_channels=96, kernel_size=2, stride=1, padding=2),
      nn.ReLU(inplace=True),
      # Layer 11: Max Pooling
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    # Block 4 (Layers 12-14)
    self.block4 = nn.Sequential(
      # Layer 12-13: Convolution (24 filters, 6x6 kernel) + ReLU
      nn.Conv2d(in_channels=96, out_channels=24, kernel_size=6, stride=1, padding=2),
      nn.ReLU(inplace=True),
      # Layer 14: Max Pooling
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    # Block 5 (Layers 15-17)
    self.block5 = nn.Sequential(
      # Layer 15-16: Convolution (24 filters, 6x6 kernel) + ReLU
      nn.Conv2d(in_channels=24, out_channels=24, kernel_size=6, stride=1, padding=2),
      nn.ReLU(inplace=True),
      # Layer 17: Batch Normalization
      nn.BatchNorm2d(24)
    )
      
    # Layers 18-22: Classification Head
    self.classifier = nn.Sequential(
      nn.Flatten(),
      # Using LazyLinear so PyTorch automatically calculates the input size 
      # after all the convolutions and pooling layers shrink the image.
      nn.LazyLinear(out_features=512), 
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.30), # 30% Dropout to prevent overfitting
      nn.Linear(in_features=512, out_features=num_classes)
      # Note: We omit the final Softmax layer here because PyTorch's 
      # CrossEntropyLoss function applies it automatically during training.
    )

  def forward(self, x):
    # Pass the image through the convolutional blocks
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    
    # Pass the flattened features through the dense layers to get predictions
    x = self.classifier(x)
    return x

# Quick test to ensure the dimensions work
if __name__ == "__main__":
  # Simulate a single batch of 1 image, with 3 color channels, sized 227x227
  dummy_input = torch.randn(1, 3, 227, 227)
  model = BrainTumorCNN(num_classes=2)
  output = model(dummy_input)
  print(f"Model initialized successfully. Output shape: {output.shape}")