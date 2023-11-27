import torch
from torch import nn

# everything in pytorch is a nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                dtype=torch.float),
                                    requires_grad=True)
        # require_grad allows us to update the value with gradient descent
        self.bias = nn.Parameter(torch.randn(1,
                                             dtype=torch.float),
                                 requires_grad=True)
    
    # forward will define our model's computation    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias;
