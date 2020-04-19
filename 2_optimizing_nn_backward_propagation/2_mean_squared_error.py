import numpy as np
from sklearn.metrics import mean_squared_error

#input
input_data = np.array([[0, 3], [1, 2], [-1, -2], [4, 0]])

# actual targets
target_actuals = [1, 3, 5, 7]

# first set of weights
weights_0 = {
    'node_0': [2, 1],
    'node_1': [1, 2],
    'output': [1, 1]
}

# sencond set of weights
weights_1 = {
    'node_0': [2, 1],
    'node_1': [1, 1.5],
    'output': [1, 1.5]
}

def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_output = (input_data_row * weights['node_0']).sum()

    # Calculate node 1 value
    node_1_output = (input_data_row * weights['node_1']).sum()

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    model_output = (hidden_layer_outputs * weights['output']).sum()
    
    # Return model output
    return(model_output)

# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)
