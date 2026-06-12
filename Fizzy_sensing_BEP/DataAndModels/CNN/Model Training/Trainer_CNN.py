import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from datac import GestureDataset
from datac import LABELS

from Trainsettings import INPUT_AMOUNT, CLASSES_AMOUNT
# ===== DIAGNOSTICS =====
print(f'Python: {sys.executable}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== DATA =====
# USE_ALL_DATA = True  # True -> train on 100% of data (no held-out test set)

full_dataset = GestureDataset()


train_data = full_dataset
# if USE_ALL_DATA:
#     # Train on 100% of the data; evaluate on the same data (no separate test set)
#     train_data = full_dataset
#     test_data = full_dataset
# else:
#     # 80/20 train/test split
#     n_test = int(0.2 * len(full_dataset))
#     n_train = len(full_dataset) - n_test
#     train_data, test_data = random_split(full_dataset, [n_train, n_test])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

#Amount of epochs used for training
EPOCHS = 100

# ===== MODEL =====
class IMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(INPUT_AMOUNT, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, CLASSES_AMOUNT) #MAX POOL 4
        # self.fc = nn.Linear(256, CLASSES_AMOUNT) #MAX POOL 2


#Met batchnorm


    def forward(self, x):
        # in:  (batch, 6, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 4)
        # out: (batch, 16, 32)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 4)
        # out: (batch, 32, 16)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool1d(x, 4)
        # out: (batch, 32, 8)

        x = torch.flatten(x, 1)
        # out: (batch, 256)

        x = self.dropout(x)
        x = self.fc(x)
        # out: (batch, 4)  — one logit per class
        return x


net = IMUNet().to(device)
print(f'Model on: {next(net.parameters()).device}')

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# ===== TRAINING =====
for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch}: loss={running_loss/len(train_loader):.4f}, time={time.time()-t0:.1f}s')

# ===== SAVE MODEL =====
# The model is saved as 'Trained_xx.pth' in the 'Models' folder, which sits one
# level up from this 'Model Training' folder in the CNN_final root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(SCRIPT_DIR, '..', 'Models', 'Trained_xx.pth')
torch.save(net.state_dict(), filename)
print(f'Saved to {filename}')

#===== EVALUATION =====

# CONFIDENCE_THRESHOLD = 0.90   # only count a prediction if it's at least 90% sure

# net.eval()
# correct = 0
# total = 0
# confident = 0      # how many predictions cleared the threshold
# unsure = 0         # how many we threw out as "not sure enough"

# all_true = []      # every true label (for the confusion matrix)
# all_pred = []      # every predicted label, regardless of confidence

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(inputs)

#         # Convert raw logits to probabilities (each row sums to 1)
#         probs = F.softmax(outputs, dim=1)

#         # Get both the top probability AND which class it belongs to
#         max_probs, predicted = torch.max(probs, dim=1)

#         # Build a True/False mask: True where confidence ≥ threshold
#         is_confident = max_probs >= CONFIDENCE_THRESHOLD

#         # Tally
#         total     += labels.size(0)
#         confident += is_confident.sum().item()
#         unsure    += (~is_confident).sum().item()

#         # Only count something as "correct" if (a) we were confident AND (b) we got it right
#         correct += ((predicted == labels) & is_confident).sum().item()

#         # Keep the raw predictions/labels for the confusion matrix
#         all_true.extend(labels.cpu().tolist())
#         all_pred.extend(predicted.cpu().tolist())

# print(f'Total windows tested:       {total}')
# print(f'Confident predictions:      {confident}  ({100*confident/total:.1f}%)')
# print(f'Marked as unsure:           {unsure}  ({100*unsure/total:.1f}%)')
# print(f'Correct AND confident:      {correct}')
# if confident > 0:
#     print(f'Accuracy among confident:   {100*correct/confident:.2f}%')
# print(f'Accuracy over all windows:  {100*correct/total:.2f}%')

# # ===== CONFUSION MATRIX =====
# # Class names in label order (0, 1, 2, ...) taken from the active LABELS map.
# class_names = [name for name, _ in sorted(LABELS.items(), key=lambda kv: kv[1])]
# labels_idx = list(range(len(class_names)))

# cm = confusion_matrix(all_true, all_pred, labels=labels_idx)

# # Print it as text (rows = true class, columns = predicted class)
# print('\nConfusion matrix (rows = true, cols = predicted):')
# header = 'true\\pred'.ljust(10) + ''.join(n[:8].rjust(9) for n in class_names)
# print(header)
# for i, name in enumerate(class_names):
#     row = name[:9].ljust(10) + ''.join(str(v).rjust(9) for v in cm[i])
#     print(row)

# # Per-class precision / recall / f1
# print('\nClassification report:')
# print(classification_report(all_true, all_pred, labels=labels_idx,
#                             target_names=class_names, zero_division=0))

# # Save a plotted version next to the model checkpoint
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# fig, ax = plt.subplots(figsize=(7, 6))
# disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=True)
# ax.set_title('Confusion matrix (test set)')
# fig.tight_layout()
# cm_filename = filename.replace('.pth', '_confusion.png')
# fig.savefig(cm_filename, dpi=150)
# print(f'Saved confusion matrix plot to {cm_filename}')