from abc import ABC, abstractmethod


class BaseLearner(ABC):
    @abstractmethod
    def perform_training(self, train_loader, num_epochs, learning_rate, weights, adversarial_training): pass

    @abstractmethod
    def compute_incorrect_array(self, train_loader): pass

    """
    Return 1d numpy array of predictions. Where each entry in the array corresponds to the prediction for a test image
    """

    @abstractmethod
    def predict_test_set(self, test_dataset): pass

    @abstractmethod
    def return_accuracy(self, data_loader): pass
