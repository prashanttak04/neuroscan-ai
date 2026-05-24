import torch
import torch.nn as nn

class BrainTumorCNN(nn.Module):
  def __init__(self, num_classes=2):
    super(BrainTumorCNN, self).__init__()
    
    self.block1 = nn.Sequential(
      # Layer 1-3:
      nn.Conv2d(in_channels=3, out_channels=128, kernel_size=6, stride=4, padding=0),
      nn.ReLU(inplace=True),

      # Layer 4: Cross Channel Normalization (Local Response Norm)
      nn.LocalResponseNorm(size=5),

      # Layer 5:
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    self.block2 = nn.Sequential(
      # Layer 6-7:
      nn.Conv2d(in_channels=128, out_channels=96, kernel_size=6, stride=1, padding=2),
      nn.ReLU(inplace=True),

      # Layer 8:
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    self.block3 = nn.Sequential(
      # Layer 9-10:
      nn.Conv2d(in_channels=96, out_channels=96, kernel_size=2, stride=1, padding=2),
      nn.ReLU(inplace=True),

      # Layer 11:
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    self.block4 = nn.Sequential(
      # Layer 12-13:
      nn.Conv2d(in_channels=96, out_channels=24, kernel_size=6, stride=1, padding=2),
      nn.ReLU(inplace=True),

      # Layer 14:
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    self.block5 = nn.Sequential(
      # Layer 15-16:
      nn.Conv2d(in_channels=24, out_channels=24, kernel_size=6, stride=1, padding=2),
      nn.ReLU(inplace=True),

      # Layer 17:
      nn.BatchNorm2d(24)
    )
      
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(out_features=512), 
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.30),
      nn.Linear(in_features=512, out_features=num_classes)

      # Note: We omit the final Softmax layer here!
    )

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    
    x = self.classifier(x)
    return x

if __name__ == "__main__":
  dummy_input = torch.randn(1, 3, 227, 227)
  model = BrainTumorCNN(num_classes=2)
  output = model(dummy_input)
  print(f"Model initialized successfully. Output shape: {output.shape}")