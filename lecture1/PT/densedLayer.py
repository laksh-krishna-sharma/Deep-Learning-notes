import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        # Initialize weights and bias
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad=True))

    def forward(self, inputs):
        # Forward propagate the inputs
        z = torch.matmul(inputs, self.W) + self.b
        # Apply non-linear activation
        output = torch.sigmoid(z)
        return output

# Test the implementation
batch_size = 4
input_dim = 3
output_dim = 2

# Create a random input tensor
inputs = torch.randn(batch_size, input_dim)

# Instantiate the layer
dense_layer = DenseLayer(input_dim, output_dim)

# Perform a forward pass
output = dense_layer(inputs)

# Print input and output
print("Input Tensor:\n", inputs)
print("\nOutput Tensor:\n", output)

