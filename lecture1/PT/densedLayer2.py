import torch as tc

layer = tc.nn.Linear(in_features=3, out_features=2)

inputs = tc.randn(5, 3)  # Batch of 5 samples, each with 3 features
output = layer(inputs)
print(output)