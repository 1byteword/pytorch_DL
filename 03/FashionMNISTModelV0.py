from torch import nn

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=input_shape, out_features=hidden_units),
                nn.Linear(in_features=hidden_units, out_features=output_shape)
            )
    
    def forward(self, x):
        return self.layer_stack(x);
