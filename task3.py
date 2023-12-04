import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian radial basis function
def gaussian_rbf(x, c, r):
    return np.exp(-(x - c)**2 / (2 * r**2))

# Generate input values
x = np.arange(0.1, 1.05, 1/22)

# Calculate desired output values
y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2

# Manually select c1, r1 for the first RBF
c1 = 0.5
r1 = 0.2

# Manually select c2, r2 for the second RBF
c2 = 0.3
r2 = 0.3

# Calculate the RBF outputs
rbf1 = gaussian_rbf(x, c1, r1)
rbf2 = gaussian_rbf(x, c2, r2)

# Create the input matrix for the perceptron training algorithm
X = np.column_stack((rbf1, rbf2, np.ones_like(x)))

# Initialize the weights
w = np.random.rand(3)

# Perform the perceptron training algorithm
epochs = 1000000
learning_rate = 0.1

for _ in range(epochs):
    print(_)
    for i in range(len(x)):
        output = np.dot(w, X[i])
        error = y[i] - output
        w += learning_rate * error * X[i]

# Print the final result
print("Final weights:", w)

# Calculate the predicted outputs
predicted_outputs = np.dot(X, w)

# Plot the predicted outputs against the actual values
plt.plot(x, y, label='Actual')
plt.plot(x, predicted_outputs, label='Predicted')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Actual vs Predicted Outputs')
plt.legend()
plt.show()
