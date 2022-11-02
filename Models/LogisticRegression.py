import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class LogisticRegressionTrainer:
    def __init__(
        self,
        num_features: int,
        learning_rate: float = 1e-2,
        num_epochs: int = 5000,
        lambd: float = 0.001,
    ) -> None:
        """Initialize a logisitc regression trainer."""
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_features = num_features
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.test_loss = None
        self.test_acc = None

        self.theta = np.zeros(self.num_features + 1)

    def gradient_descent_step(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Perform a single step of gradient update.

        Args:
            x: A matrix of features.
            y: A vector of labels.
        """

        alpha = self.learning_rate
        self.theta = self.theta - (alpha * self.cross_entropy_loss_derivative(x, y))

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Convert raw model output (logits) to probabilities.

        Args:
            z: Raw model output (logits).

        Returns:
            A vector (or float, if your input is a scalar) of probabilties.
        """

        return (1/(1 + np.exp(-z)))

    def cross_entropy_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates the binary cross-entropy loss given predictions and targets.
        The loss function should add the regularization term.

        Args:
            pred: Predicted labels (probabilities).
            target: Ground-truth labels.

        Returns:
            A scalar of loss.
        """
        assert pred.shape == target.shape
        m = target.size
        #print(pred)
        #print(target)
        answer = (-1 * 1/m * np.sum(target * np.log(pred) + (1 - target) * np.log(1-pred))) + (self.lambd * np.sum(self.theta[1:] ** 2))
        #print(answer)
        return(answer)

    def cross_entropy_loss_derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the loss function w.r.t. theta. The derivative of the
        loss function should also add the derivative of the L2 regularization term.

        Args:
            x: Feature vectors.
            y: Ground-truth labels.

        Returns:
            A vector with the same dimension as theta, where each element is the
            partial derivative of the loss function w.r.t. the corresponding element
            in theta.
        """

        pred = self.sigmoid(np.dot(x, self.theta))
        m = y.size
        #print(x)
        #print(y)
        #print(pred)
        #print("Loss Derivative")
        newTheta = self.theta[1:3]
        answer = (1/m * (np.sum((pred - y)[:, None] * x, axis  = 0))) + (self.lambd * np.sum(2 * self.theta[1:]))
        #print(answer)
        return (answer)


    def accuracy(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates the percentage of matched labels given predictions and targets.

        Args:
            pred: Predicted labels (rounded probabilities).
            target: Ground-truth labels.

        Return:
            The accuracy score (a float) given the predicted labels and the true labels.
        """
        assert pred.shape == target.shape

        predRound = np.round(pred)
        correct = np.sum(predRound == target)
        return correct/np.size(pred)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Run gradient descent for n epochs, where n = self.num_epochs. In every epoch,
            1. Update theta.
            2. Calculate the training loss & accuracy given the current theta, and append 
               then to self.train_loss_history and self.train_acc_history.
            3. Calculate the validation loss & accuracy given the current theta, and 
               append then to self.train_loss_history and self.train_acc_history.

        If you wish to use the bias trick, please remember to use it before the for loop.

        Args:
            x_train: Feature vectors for training.
            y_train: Ground-truth labels for training.
            x_val: Feature vectors for validation.
            y_val: Ground-truth labels for validation.
        """

        x_train = np.hstack((np.ones((len(x_train),1)), x_train))
        x_val = np.hstack((np.ones((len(x_val),1)), x_val))
        print(x_train)
        
        for i in range(self.num_epochs):
            #print(self.theta)
            hypo_train = self.sigmoid(np.dot(x_train, self.theta))
            hypo_val = self.sigmoid(np.dot(x_val, self.theta))
            #print(hypo_train)
            '''
            print("Hypo sizes")
            print(hypo_train.shape)
            print(hypo_val.shape)
            print(y_train.shape)
            print(y_val.shape)
            '''
            #print(self.cross_entropy_loss(hypo_train, y_train))
            self.train_loss_history.append(self.cross_entropy_loss(hypo_train, y_train))
            self.val_loss_history.append(self.cross_entropy_loss(hypo_val, y_val))
            self.train_acc_history.append(self.accuracy(hypo_train, y_train))
            self.val_acc_history.append(self.accuracy(hypo_val, y_val))
            self.gradient_descent_step(x_train, y_train)

        print(self.theta)


    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the model on test set and store the test loss int self.test_loss and 
        test accuracy in self.test_acc. In other words, you should get the test loss and accraucy here.

        If you used the bias trick in train(), you have to also use it here.

        Args:
            x_test: Feature vectors for testing.
            y_test: Ground-truth labels for testing.
        """

        x_test = np.hstack(( np.ones((len(x_test),1)), x_test))
        hypo_test = self.sigmoid(np.dot(x_test, self.theta))
        self.test_loss = self.cross_entropy_loss(hypo_test, y_test)
        self.test_acc = self.accuracy(hypo_test, y_test)

        