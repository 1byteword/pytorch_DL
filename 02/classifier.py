from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from CircleModelV0 import CircleModelV0
from CircleModelV1 import CircleModelV1
from CircleModelV2 import CircleModelV2

from torch import nn
from pathlib import Path
from helper_functions import plot_predictions, plot_decision_boundary
import pandas as pd
import matplotlib.pyplot as plt
import torch
import requests

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(f"First X features:\n{X[:5]}")
print(f"\nFirst y labels:\n{y[:5]}")



circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

#print(circles.head(10))

#print(circles.label.value_counts())

#plt.scatter(x=X[:, 0],
#            y=X[:, 1],
#            c=y,
#            cmap=plt.cm.RdYlBu);

#plt.show()


X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#print(X[:5])
#print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

#print(len(X_train))
#print(len(X_test))
#print(len(y_train))
#print(len(y_test))

model_3 = CircleModelV0()
print(model_3)

model_3 = nn.Sequential(
        nn.Linear(in_features=2, out_features=5),
        nn.Linear(in_features=5, out_features=1)
        )

print(model_3)


untrained_preds = model_3(X_test)
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc;

def sigmoid(x):
    return 1 / (1 + torch.exp(-x));

y_logits = model_3(X_test)[:5]
print(y_logits)

y_pred_probs = torch.sigmoid(y_logits)

print(y_pred_probs)

y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_3(X_test))[:5])

print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

y_preds.squeeze()

print(y_test[:5])

torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    model_3.train()

    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,
                   y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    model_3.eval()

    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss : {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss: .5f}% | Test acc: {test_acc:.2f}%")


#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title("Train")
#plot_decision_boundary(model_3, X_train, y_train)
#plt.subplot(1, 2, 2)
#plt.title("Test")
#plot_decision_boundary(model_3, X_test, y_test)
#plt.show()

model_1 = CircleModelV1()
print(model_1)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)

torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()

    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,
                            y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}%, Test acc: {test_acc:.2f}%")


plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()


weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

print(len(X_regression))
print(X_regression[:5], y_regression[:5])

X, y = make_circles(n_samples=1000,
                    noise=0.03,
                    random_state=42,
        )

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train[:5])
print(y_train[:5])


model_2 = CircleModelV2()
print(model_2)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)

torch.manual_seed(42)
epochs=1000

for epoch in range(epochs):
    y_logits = model_2(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_2, X_test, y_test) # model_3 = has non-linearity
plt.show()
