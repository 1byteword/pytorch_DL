import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from CircleModelV0 import CircleModelV0

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)


print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

print(circles.head(10))

#plt.scatter(x=X[:, 0],
#            y=X[:, 1],
#            c=y,
#            cmap=plt.cm.RdYlBu);
#plt.show()


print(X.shape)
print(y.shape)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print("----")
print(X[:5])
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


model_0 = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1)
        )

print(model_0)

untrained_preds = model_0(X_test)
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")


# loss_fn = nn.BCELoss() # BCELoss has no sigmoid layer and thus is less stable
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss has a built-in sigmoid layer

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc;

torch.manual_seed(42)
epochs = 100

y_logits = model_0(X_test)[:5]
print(y_logits)

y_pred_probs = torch.sigmoid(y_logits)
print("---------SIGMOID ACTIVATED-----------")
print(y_pred_probs)

y_preds = torch.round(y_pred_probs)

y_pred_labels = torch.round(torch.sigmoid(model_0(X_test[:5])))

print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

print(y_preds.squeeze())

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,
                   y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,
                            y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    #loss = loss_fn(y_pred, y_train)
