import torch

tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]], dtype=torch.float32)

linear = torch.nn.Linear(in_features=2, out_features=6)

x = tensor_A

output = linear(x)

print(f"Output:\n{output}\n\nOutput Shape: {output.shape}")


x = torch.arange(0, 100, 10)
print(x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Mean: {x.type(torch.float32).mean()}")
print(f"Sum: {x.sum()}")


print(f"Index where max value occurs: {x.argmax()}")
print(f"Index where min value occurs: {x.argmin()}")



tensor_int8 = tensor_A.type(torch.int8)
print(tensor_int8)


x = torch.arange(1.,8.)
print(x)
print(x.shape)

x_reshaped = x.reshape(1, 7)
print(x_reshaped)
print(x_reshaped.shape)

x_reshaped_stacked = torch.stack([x_reshaped,x_reshaped,x_reshaped], dim=0)

print(x_reshaped_stacked)


x = torch.arange(1,10).reshape(1,3,3)
print(x[:, 0])
print(x[:, :, 1])

print(x[:, 1, 1])

