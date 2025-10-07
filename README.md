# workshop-unit-2

### Building an AI Classifier: Identifying Cats, Dogs & Pandas with PyTorch :

### CODE :
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models,datasets
from torchvision.utils import make_grid

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

train_transform = transforms.Compose([
    transforms.RandomRotation(10),      # randomly rotate image +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # randomly flip left-right 50% of images
    transforms.Resize(224),             # resize shortest side to 224 pixels
    transforms.CenterCrop(224),         # crop to 224x224 at center
    transforms.ToTensor(),              # convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225]) # ImageNet std
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

root = r'C:\Users\admin\Desktop\DEEP-2\Cat-Dog_Pandas'

train_data = datasets.ImageFolder(os.path.join(root, 'Train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'Test'), transform=test_transform)

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

class_names = train_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')


```

### OUTPUT :

```python
ResNet18model = models.resnet18(pretrained=True)
ResNet18model
```

### OUTPUT :
['cat', 'dog', 'panda']
Training images available: 2100
Testing images available:  300


```python
for param in ResNet18model.parameters():
    param.requires_grad = False  # freeze all convolutional layers

num_features = ResNet18model.fc.in_features  # 512 for ResNet18

ResNet18model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 3)   # 3 classes: cat, dog, panda
)
ResNet18model
# Print number of elements in each parameter tensor
for param in ResNet18model.parameters():
    print(param.numel())
```
### OUTPUT :
9408
64
64
36864
64
64
36864
64
64
36864
64
64
36864
64
64
73728
128
128
147456
128
128
8192
128
128
147456
...
131072
256
768
3

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ResNet18model.fc.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Move the model to the device
ResNet18model = ResNet18model.to(device)

import time
import torch

# ----------------------------
# Parameters
# ----------------------------
epochs = 3 # because 3 gives better accuracy than 5
max_trn_batch = 800   # optional, can reduce for quick testing
max_tst_batch = 300

train_losses = []
test_losses = []
train_correct = []
test_correct = []

# ----------------------------
# Start timing
# ----------------------------
start_time = time.time()

# ----------------------------
# Training loop
# ----------------------------
for i in range(epochs):
    ResNet18model.train()   # set model to training mode
    trn_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        if b == max_trn_batch:
            break

        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass
        y_pred = ResNet18model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute batch accuracy
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Print interim results
        if (b+1) % 200 == 0:
            print(f'epoch: {i+1:2}  batch: {b+1:4} [{(b+1)*10:6}/2100]  '
                  f'loss: {loss.item():10.8f}  '
                  f'accuracy: {trn_corr.item()*100/((b+1)*train_loader.batch_size):7.3f}%')

    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    # ----------------------------
    # Validation loop
    # ----------------------------
    ResNet18model.eval()   # set model to evaluation mode
    tst_corr = 0

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if b == max_tst_batch:
                break

            X_test, y_test = X_test.to(device), y_test.to(device)

            y_val = ResNet18model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

        loss_val = criterion(y_val, y_test)
        test_losses.append(loss_val.item())
        test_correct.append(tst_corr.item())

print(f'\nDuration: {time.time() - start_time:.0f} seconds')
# Save best model checkpoint
best_val_acc = 0.0  # initialize best validation accuracy

val_acc = tst_corr.item() / len(test_data)  # calculate validation accuracy
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(ResNet18model.state_dict(), 'best_resnet18_model.pth')
    print(f"Saved best model at epoch {i+1} with val_acc = {val_acc*100:.2f}%")

print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# Test accuracy
print(test_correct)
print(f'Test accuracy: {test_correct[-1]*100/len(test_data):.3f}%')
image_index = 250
img, label = test_data[image_index]

# Convert tensor shape from [C,H,W] to [H,W,C] for plotting
plt.figure(figsize=(4, 4))
plt.imshow(img.permute(1, 2, 0))
plt.axis('off')
plt.title("Selected Image")
plt.show()

# Evaluate model
ResNet18model.eval()
with torch.no_grad():
    img_input = img.unsqueeze(0).to(device)
    pred = ResNet18model(img_input).argmax(dim=1)

predicted_class = class_names[pred.item()]
true_class = class_names[label]

print(f"True class: {true_class}")
print(f"Predicted class: {predicted_class}")

```

### OUTPUT :
epoch:  1  batch:  200 [  2000/2100]  loss: 0.59238672  accuracy:  87.400%
epoch:  2  batch:  200 [  2000/2100]  loss: 0.01386434  accuracy:  92.250%
epoch:  3  batch:  200 [  2000/2100]  loss: 0.00762072  accuracy:  93.150%

Duration: 89 seconds
Saved best model at epoch 3 with val_acc = 96.67%
Duration: 118 seconds

[280, 290, 290]
Test accuracy: 96.667%
<img width="444" height="523" alt="Screenshot 2025-10-07 221558" src="https://github.com/user-attachments/assets/f677a247-4d21-4613-b8c3-d72edf2493bb" />


```python
test_load_all = DataLoader(test_data, batch_size=20, shuffle=False)

all_preds = []
all_labels = []

ResNet18model.eval()
with torch.no_grad():
    for X_test, y_test in test_load_all:
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_val = ResNet18model(X_test)
        predicted = torch.max(y_val, 1)[1]

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

# Compute confusion matrix
import seaborn as sns
arr = confusion_matrix(all_labels, all_preds)

# Optional: use your class names here if available
# Example:
# class_names = ['cat', 'dog', 'panda']

df_cm = pd.DataFrame(arr, index=class_names, columns=class_names)

plt.figure(figsize=(9,6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix - ResNet18")
plt.show()
# Save model after training
torch.save(ResNet18model.state_dict(), "resnet18_catdogpanda.pth")

print("Model saved as resnet18_catdogpanda.pth")

```
### OUTPUT :

<img width="906" height="692" alt="Screenshot 2025-10-07 221609" src="https://github.com/user-attachments/assets/38d99d75-d865-4a7f-a9c4-82a5671e3b23" />

