import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feed-forward network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# Create dummy input and target
x = torch.randn(5, 10)  # batch of 5, 10 features
y = torch.tensor([0, 1, 0, 1, 0])

# Forward pass
output = model(x)
loss = loss_function(output, y)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())


# Documentation and Use Case
"""
creation and training of a simple feed-forward neural network using PyTorch.
PyTorch is a popular deep learning framework that provides flexibility and ease of use for building and training neural networks.

Use Case:
- The network is designed to classify input data into one of two classes.
- It uses a single fully connected layer with 10 input features and 2 output classes.
- The script includes a forward pass, loss computation, and a backward pass for optimization.

Key PyTorch Components Used:
1. `torch.nn`: Provides modules and classes for building neural networks (e.g., `nn.Linear`, `nn.CrossEntropyLoss`).
2. `torch.optim`: Contains optimization algorithms (e.g., `optim.Adam`).
3. `torch.Tensor`: Represents multi-dimensional arrays for input data and model parameters.
"""

# Example of how to use the model
def train_simple_nn():
  """
  Function to demonstrate training of the SimpleNN model on dummy data.
  """
  # Initialize the model, optimizer, and loss function
  model = SimpleNN()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  loss_function = nn.CrossEntropyLoss()

  # Dummy input and target
  x = torch.randn(5, 10)  # batch of 5, 10 features
  y = torch.tensor([0, 1, 0, 1, 0])  # target labels

  # Training loop (single iteration for demonstration)
  model.train()  # Set the model to training mode
  optimizer.zero_grad()  # Clear gradients
  output = model(x)  # Forward pass
  loss = loss_function(output, y)  # Compute loss
  loss.backward()  # Backward pass
  optimizer.step()  # Update parameters

  print("Training completed. Final Loss:", loss.item())

# Call the training function
train_simple_nn()