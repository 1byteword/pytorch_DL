import torch
import matplotlib.pyplot as plt
from torch import nn
from LinearRegressionModel import LinearRegressionModel

covering = { 1: "data (prepare and load)",
             2: "build model",
             3: "fitting the model to data (training)",
             4: "making predictions and evaluating a model (inference)",
             5: "saving and loading a model",
             6: "finishing touches" }

print(torch.__version__)

# some known parameters
weight = 0.7
bias = 0.3

# create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X)

# print out the last ten elements of each array
print(X[:10])
print(y[:10])
print("------------------\n")



train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

torch.manual_seed(42)
model_0 = LinearRegressionModel()

list(model_0.parameters())

print(model_0.state_dict())


with torch.inference_mode():
    y_preds = model_0(X_test)


# checking predictions...
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")




torch.manual_seed(42)
# mount
loss = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 100

train_loss_values = []
test_loss_values = []
epoch_count = []

# train
for epoch in range(epochs):
    model_0.train()
    
    y_pred = model_0(X_train)

    loss_amt = loss(y_pred, y_train)

    optimizer.zero_grad()

    loss_amt.backward()

    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        test_prediction = model_0(X_test)
        test_loss = loss(test_prediction, y_test.type(torch.float))

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss_amt.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss_amt} | MAE Test Loss {test_loss} ")

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")
