import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import trange


class LinearRegression:

    def __init__(self,
                 num_features: int,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 5000) -> None:

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_features = num_features
        self.train_loss_history = []
        self.val_loss_history = []
        self.test_loss = None
        self.theta = np.zeros(self.num_features + 1)

    def gradient_descent_step(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Perform a single step of gradient update on self.theta.

        Args:
            x: A matrix of features.
            y: A vector of labels.
        """
        alpha = self.learning_rate
        self.theta = self.theta - (alpha * self.mse_loss_derivative(x, y))

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Run gradient descent for n epochs, where n = self.num_epochs. In every epoch,
            1. Calculate the training loss given the current theta, and append it to
               self.train_loss_history.
            2. Calculate the validation loss given the current theta, and append it to
               self.val_loss_history.
            3. Update theta.

        Args:
            x: A matrix of features.
            y: A vector of labels.
        """
        # ========== YOUR CODE STARTS HERE ==========

        for i in range(self.num_epochs):
            hypo_train = np.dot(x_train, self.theta)
            hypo_val = np.dot(x_val, self.theta)
            self.train_loss_history.append(
                LinearRegressionTrainer.mse_loss(hypo_train, y_train))
            self.val_loss_history.append(
                LinearRegressionTrainer.mse_loss(hypo_val, y_val))
            self.gradient_descent_step(x_train, y_train)

    def mse_loss_derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the loss function w.r.t. theta.

        Args:
            x: A matrix of features.
            y: A vector of labels.

        Returns:
            A vector with the same dimension as theta, where each element is the
            partial derivative of the loss function w.r.t. the corresponding element
            in theta.
        """
        m = y.size
        J = (1 / m) * np.sum(
            ((np.dot(x, self.theta) - y)[:, None] * x), axis=0)

        return J

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the model on test set and store the test loss int self.test_loss.

        Args:
            x_test: A matrix of features.
            y_test: A vector of labels.
        """
        hypo_test = np.dot(x_test, self.theta)
        self.test_loss = LinearRegressionTrainer.mse_loss(hypo_test, y_test)

    def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the mean squared error given prediction and target. 

        Args:
            pred: A vector of predictions.
            target: A vector of labels.

        Returns:
            Mean squared error between each element in pred and target.
        """
        assert pred.shape == target.shape
        n = target.size
        answer = sum((pred - target)**2) / (2 * n)
        return answer

    def plot_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Plot a dataset with 2-d feature vectors and binary labels. 

        Args:
            x: 2-d feature vectors
            y: 1-d binary labels.
        """
        class0_idx = np.where(y == 0)[0]
        class1_idx = np.where(y == 1)[0]
        feature0 = x[:, 0]
        feature1 = x[:, 1]
        plt.scatter(feature0[class0_idx], feature1[class0_idx], label="0")
        plt.scatter(feature0[class1_idx], feature1[class1_idx], label="1")
        plt.legend()
        plt.show()