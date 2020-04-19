import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

n_updates = 20
mse_hist = []
learning_rate = 0.01

def get_slope(input_data, target, weights):
    # Calculate the predictions: preds
    preds = (input_data * weights).sum()

    # Calculate the error: error
    error = preds - target

    # Calculate the slope: slope
    slope = 2 * input_data * error

    return slope

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - learning_rate * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()