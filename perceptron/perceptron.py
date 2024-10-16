import numpy as np

class Perceptron:
    """
    A simple Perceptron implementation.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Perceptron model on the training data.

        Args:
            X: Training data (features). Shape: (n_samples, n_features)
            y: Target values (labels). Shape: (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to {-1, 1} for Perceptron update rule
        y_ = np.where(y > 0, 1, -1)

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Update weights and bias if prediction is incorrect
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Data to make predictions on. Shape: (n_samples, n_features)

        Returns:
            Predicted labels. Shape: (n_samples,)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        """
        The unit step activation function.
        """
        return np.where(x >= 0, 1, -1)

# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
    y = np.array([0, 1, 0, 1])

    # Create and train the Perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iters=100)
    perceptron.fit(X, y)

    # Make predictions
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)
