import matplotlib.pyplot as plt

def visualize_risk(predictions, actuals):
    # Convert tensors to numpy for plotting
    predictions = predictions.numpy().flatten()
    actuals = actuals.numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predicted Threat Level", color="blue", alpha=0.7)
    plt.plot(actuals, label="Actual Threat Level", color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Test Sample")
    plt.ylabel("Threat Level")
    plt.legend()
    plt.title("Predicted vs. Actual Threat Level")
    plt.show()
