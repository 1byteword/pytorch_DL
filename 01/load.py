import torch

from torch import nn
from LinearRegressionModel import LinearRegressionModel


loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
