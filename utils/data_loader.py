import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_data(data_path, test_size=0.2):
    # Load data from CSV
    data = pd.read_csv(data_path)
    
    # Define features and target
    features = data[["uav_altitude", "uav_speed", "uav_direction", "missile_distance", 
                     "missile_speed", "missile_angle_of_approach", "temperature", 
                     "wind_speed", "pressure"]]
    target = data["threat_level"]
    
    # Convert to tensors
    X = torch.tensor(features.values, dtype=torch.float32)
    y = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)  # Ensuring y is 2D

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test
