import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

import matplotlib.pyplot as plt


class PropertyPriceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # convert TimeTo.. values
        def extract_time(s):
            matches = re.findall(r'\d+', s)
            if len(matches) == 2:
                return (int(matches[0]) + int(matches[1])) / 2
            else:
                return None

        self.data['TimeToBusStop'] = self.data['TimeToBusStop'].apply(extract_time)
        self.data['TimeToSubway'] = self.data['TimeToSubway'].apply(extract_time)

        # Encode categorical variables
        le = LabelEncoder()
        self.data['HallwayType'] = le.fit_transform(self.data['HallwayType'])
        self.data['HeatingType'] = le.fit_transform(self.data['HeatingType'])
        self.data['AptManageType'] = le.fit_transform(self.data['AptManageType'])
        self.data['SubwayStation'] = le.fit_transform(self.data['SubwayStation'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.data.iloc[idx, 1:].values.astype(np.float32)
        sale_price = self.data.iloc[idx, 0]
        label = 0 if sale_price < 100000 else 1 if sale_price < 350000 else 2
        return torch.tensor(features), torch.tensor(label)


class PropertyPriceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PropertyPriceClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(loss.item())

        epoch_loss = running_loss / len(dataloader)
        print(epoch_loss)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    return epoch_losses


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def main():
    # Hyperparameters
    input_size = 16
    hidden_size = 32
    num_classes = 3
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Load dataset
    training_data = PropertyPriceDataset("/home/nkvch/studia/SSNE/miniprojekt2/train_data.csv")

    train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, criterion, and optimizer
    model = PropertyPriceClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    losses = train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    plot_losses(losses)


if __name__ == "__main__":
    main()