# from abc import ABC, abstractmethod
#
#
# class BaseLearner(ABC):
#     @abstractmethod
#     @staticmethod
#     def perform_training(model, device, train_loader, num_epochs, learning_rate, weights, adversarial_training): pass
#
#     @abstractmethod
#     @staticmethod
#     def compute_incorrect_array(model, device, train_loader): pass
#
#     """
#     Return 1d numpy array of predictions. Where each entry in the array corresponds to the prediction for a test image
#     """
#     @abstractmethod
#     @staticmethod
#     def predict_test_set(model, device, test_dataset): pass
#
#     @abstractmethod
#     @staticmethod
#     def return_accuracy(model, device, data_loader): pass
