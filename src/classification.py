import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(400000, 4096)
        self.fc2 = nn.Linear(4096 + 4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 2)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)

    def forward(self, x):
        x1 = x[:, :400000]
        x2 = x[:, 400000:]
        
        x1 = torch.relu(self.fc1(x1))
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x_combined = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.fc2(x_combined))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        output = self.fc6(x)
        
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare features
with open('hidden_states/train_data.pkl', 'rb') as f:
    data_embeddings_train_2 = np.array(pickle.load(f))[:, 14, :]
with open('hidden_states/train_instruction.pkl', 'rb') as f:
    instruction_embeddings_train_2 = np.array(pickle.load(f))[:, 14, :]
with open('gradients/train_data_13.pkl', 'rb') as f:
    data_embeddings_train_1 = pickle.load(f)
with open('gradients/train_instruction_13.pkl', 'rb') as f:
    instruction_embeddings_train_1 = pickle.load(f)
data_embeddings_train = np.concatenate((data_embeddings_train_1, data_embeddings_train_2), axis=1)
instruction_embeddings_train = np.concatenate((instruction_embeddings_train_1, instruction_embeddings_train_2), axis=1)
x_train = np.concatenate((data_embeddings_train, instruction_embeddings_train), axis=0)
y_train = np.array([0] * len(data_embeddings_train) + [1] * len(instruction_embeddings_train))
x_train, y_train = shuffle(x_train, y_train, random_state=42)

# Train
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
model = CustomMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
batch_size = 64
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
epochs = 500
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')
    if running_loss / len(train_loader) <= 0.0001 and epoch > 80:
        break
torch.save(model.state_dict(), 'combine.pth')

# Test
with open('hidden_states/test_data1.pkl', 'rb') as f:
    data_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('hidden_states/test_instruction1.pkl', 'rb') as f:
    instruction_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('gradients/test_data1_13.pkl', 'rb') as f:
    data_embeddings_test_1 = pickle.load(f)
with open('gradients/test_instruction1_13.pkl', 'rb') as f:
    instruction_embeddings_test_1 = pickle.load(f)
data_embeddings_test = np.concatenate((data_embeddings_test_1, data_embeddings_test_2), axis=1)
instruction_embeddings_test = np.concatenate((instruction_embeddings_test_1, instruction_embeddings_test_2), axis=1)
x_test = np.concatenate((data_embeddings_test, instruction_embeddings_test), axis=0)
y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, y_pred = torch.max(outputs, 1)
y_pred = y_pred.cpu().numpy()
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy1: {accuracy * 100:.2f}%")


with open('hidden_states/test_data2.pkl', 'rb') as f:
    data_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('hidden_states/test_instruction2.pkl', 'rb') as f:
    instruction_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('gradients/test_data2_13.pkl', 'rb') as f:
    data_embeddings_test_1 = pickle.load(f)
with open('gradients/test_instruction2_13.pkl', 'rb') as f:
    instruction_embeddings_test_1 = pickle.load(f)
data_embeddings_test = np.concatenate((data_embeddings_test_1, data_embeddings_test_2), axis=1)
instruction_embeddings_test = np.concatenate((instruction_embeddings_test_1, instruction_embeddings_test_2), axis=1)
x_test = np.concatenate((data_embeddings_test, instruction_embeddings_test), axis=0)
y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, y_pred = torch.max(outputs, 1)
y_pred = y_pred.cpu().numpy()
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy2: {accuracy * 100:.2f}%")


with open('hidden_states/test_data3.pkl', 'rb') as f:
    data_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('hidden_states/test_instruction3.pkl', 'rb') as f:
    instruction_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('gradients/test_data3_13.pkl', 'rb') as f:
    data_embeddings_test_1 = pickle.load(f)
with open('gradients/test_instruction3_13.pkl', 'rb') as f:
    instruction_embeddings_test_1 = pickle.load(f)
data_embeddings_test = np.concatenate((data_embeddings_test_1, data_embeddings_test_2), axis=1)
instruction_embeddings_test = np.concatenate((instruction_embeddings_test_1, instruction_embeddings_test_2), axis=1)
x_test = np.concatenate((data_embeddings_test, instruction_embeddings_test), axis=0)
y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, y_pred = torch.max(outputs, 1)
y_pred = y_pred.cpu().numpy()
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy3: {accuracy * 100:.2f}%")


with open('hidden_states/test_data4.pkl', 'rb') as f:
    data_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('hidden_states/test_instruction4.pkl', 'rb') as f:
    instruction_embeddings_test_2 = np.array(pickle.load(f))[:, 14, :]
with open('gradients/test_data4_13.pkl', 'rb') as f:
    data_embeddings_test_1 = pickle.load(f)
with open('gradients/test_instruction4_13.pkl', 'rb') as f:
    instruction_embeddings_test_1 = pickle.load(f)
data_embeddings_test = np.concatenate((data_embeddings_test_1, data_embeddings_test_2), axis=1)
instruction_embeddings_test = np.concatenate((instruction_embeddings_test_1, instruction_embeddings_test_2), axis=1)
x_test = np.concatenate((data_embeddings_test, instruction_embeddings_test), axis=0)
y_test = np.array([0] * len(data_embeddings_test) + [1] * len(instruction_embeddings_test))
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, y_pred = torch.max(outputs, 1)
y_pred = y_pred.cpu().numpy()
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy4: {accuracy * 100:.2f}%")
