from utils.data_loader import load_data
import torch
from models.model import ThreatEstimationNet
from models.train import train_model
from utils.visualize import visualize_risk

def main():
    # Step 1: Train the model
    train_model()

    # Step 2: Load trained model and test data
    X_train, X_test, y_train, y_test = load_data("synthetic_data.csv")
    model = ThreatEstimationNet(input_dim=X_train.shape[1])
    model.load_state_dict(torch.load("models/threat_estimation_model.pth"))

    # Step 3: Visualize predictions for test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        visualize_risk(predictions, y_test)

if __name__ == "__main__":
    main()
