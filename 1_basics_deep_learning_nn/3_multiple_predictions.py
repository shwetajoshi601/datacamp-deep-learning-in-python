import numpy as np

input_data = np.array([[-1, 2], [2, 3], [7, 10], [1, 1]])

weights = {
    'node_0': [3, 3],
    'node_1': [1, 5],
    'output': [2, -1]
}

def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    # returns 0 if the number is negative, returns the number if it is positive
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)
        