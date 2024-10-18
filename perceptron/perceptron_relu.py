import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.num_epochs):
            for i in range(num_samples):
                x = X[i]
                y_pred = self.predict(x)
                error = y[i] - y_pred

                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.maximum(0, z)  # ReLU activation

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

perceptron = Perceptron()
perceptron.fit(X, y)

# Make predictions
new_data = np.array([[0.5, 0.8]])
predictions = perceptron.predict(new_data)
print(predictions)