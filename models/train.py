import torch
import torch.optim as optim
import torch.nn as nn
from models.model import ThreatEstimationNet
from utils.data_loader import load_data


def train_model(data_path="synthetic_data.csv", num_epochs=100):
    X_train, X_test, y_train, y_test = load_data(data_path)
    input_dim = X_train.shape[1]
    
    model = ThreatEstimationNet(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), "models/threat_estimation_model.pth")

if __name__ == "__main__":
    train_model()
