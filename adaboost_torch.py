import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import confusion_matrix as CM
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from CNN import ConvNet
from MLP import MLP

from MyDataset import CustomDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON = 1e-10

class AdaBoost(object):
    """
    Adaboost constructor.
    @param m - number of learners Adaboost will use in ensembling
    @param X - training data X, of length n. NOTE we assume this training data has gone through all necessary
        transformations, and is thus is directly able to be used for training.
    @param y - training labels corresponding to data, of length n
    @param independent_variable - if performing experiments, which variable is being changed
    @param independent_variable_values - if performing experiments, pass an array of all the potential values that you want
        to try
    """
    def __init__(self, m, X, Y, training_data, base_learner, num_epochs, batch_size, nn_learning_rate,
                 independent_variable="ALL", independent_variable_values=[]):

        # Defining the number of learners
        self._number_learners = m
        # Defining the learning rate of the learners. Weight applied to each classifier at each boosting iteration.
        # A higher learning rate increases the contribution of each classifier. There is a trade-off between the
        # `learning_rate` and `n_estimators` parameters.
        self._adaboost_learning_rate = 1
        # Making an array for the learners (these can be pytorch CNN objects or whatever machine learning classifier)
        self._learner_array = []
        # An array that stores the "amount of say" of each learner i.e its weight in the final prediction
        self._learner_weights = np.zeros(self._number_learners)
        # An array that stores the error of each learner
        self._learner_errors = np.ones(self._number_learners)

        # Data related to training data
        self._original_X = X
        self._Y = Y
        self._training_data = training_data

        # Data related to the weak learners.
        self._base_learner = base_learner
        self._nn_learning_rate = nn_learning_rate
        self._nn_num_epoch = num_epochs
        self._nn_batch_size = batch_size

        # --------------------------------------------------------------------------------------------------------
        # The following are variables related to performing mini-experiments (including the independent variables)
        # --------------------------------------------------------------------------------------------------------

        # are we altering multiple independent variable at the same time (i.e testing correlations between variables)
        self._multiple_independent_variable = False
        # what current independent variable are we currently experimenting with. Default option of "ALL" means we are
        # not altering any, and using best values found
        self._chosen_independent_variable = independent_variable
        # values for the independent variables we want to test
        self._independent_variable_values = independent_variable_values
        # Parameter that decides if we are doing SAMME or SAMME.R algorithm
        self._algorithm = "SAMME"
        # Following are the independent variables, names self-explanatory
        self._nn_shape = []
        self._nn_training_loss_function = "cross-entropy-loss"
        self._nn_activation_function = "ReLU"
        self._nn_perturbation_radius = 8 / 256
        self._nn_adversary_attack_algo = "PGD"
        self._nn_optimiser = "SGD"

    def I(self, flag):
        return 1 if flag else 0

    def sign(self, x):
        return abs(x) / x if x != 0 else 1

    """
    @:param training_data: a pytorch dataset object
    @:param normal_training_weights: a normalised tensor containing probabilities for each training example
    @:param batch_size: the batch size desired for the returned train loader
    
    This method takes in a training dataset and a weight array, and uses the WeightedRandomSampler to 
    sample training examples from the training dataset. The DataLoader is used with this sampler to return a loader
    object that has split the sampled training dataset into batches. 
    """
    def draw_random_sample(self, training_data, normal_training_weights, batch_size):
        weighted_samp = WeightedRandomSampler(normal_training_weights, num_samples=len(training_data), replacement=True)
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                   sampler=weighted_samp)
        return train_loader

    def update_weights_new(self, train_loader, incorrect, alpha, w_prev, batch_size):
        w_updates = [[] for i in range(len(w_prev))]
        k = 0
        for images, labels, index in train_loader:
            # number of images/labels/indices we get == batch_size
            for i in range(batch_size):
                w_update_value = w_prev[k] * np.exp(alpha * incorrect[k])
                original_training_index = index[i]
                w_updates[original_training_index].append(w_update_value)

        # each weight is averaged
        w_new = np.zeros(len(w_prev))
        for i, lst in enumerate(w_updates):
            # if the length of update array is 0, that means that this original training example wasn't part of this \
            # iterations sampled data. We thus need to set the weight of this training example to that equal in w_prev
            if len(lst) == 0:
                # TODO alter this, how much weight should an example get when it wasn't sampled by the previous learner
                w_new[i] = w_prev[i] * 1.1
            # If the length is greater, than we take the average
            else:
                w_new[i] = sum(lst) / len(lst)

        # check for non-negative total sum or ∞ sum
        w_sum = sum(w_new)
        if w_sum <= 0:
            print("EYY Yoo, weights negative, for learner")
            raise ValueError("weights are negative")
        if not np.isfinite(w_sum):
            raise ValueError("Sample weights have reached infinite values, causing overflow. "
                             "Iterations stopped. Try lowering the learning rate.")
        # normalise
        w_new = w_new / w_sum
        return torch.tensor(w_new)

    def fit_ensemble(self):
        # We need to first do some initializations for the first ensemble.
        N = len(self._training_data)

        # The weight array stores the weight of training data x_i at index i (thus w[0] corresponds to weight of
        # the very first training data
        w = torch.tensor([1 / N for i in range(N)]).to(device)
        y = self._Y

        #self._classes = np.array(sorted(list(set(y))))
        #num_classes = len(self._classes)
        #self._num_classes = num_classes
        num_classes = 10
        self._num_classes = num_classes

        for m in range(self._number_learners):
            # Sample data based on weight distribution (we don't if m == 0 i.e. we are training the first learner)
            if m >= 1:
                train_loader = self.draw_random_sample(self._training_data, w, self._nn_batch_size)
            else:
                train_loader = torch.utils.data.DataLoader(self._training_data, batch_size=self._nn_batch_size,
                                                           shuffle=True)
                idx = [i for i in range(N)]

            # Train "weak" learner based on sample training data
            # TODO, when i make this non-hardcoded, i need to add randomness with seeds for the base classifier
            # TODO / for example, see scikit impl or Adaboost-CNN code
            print("STARTED TRAINING LEARNER", m)
            weak_learner = MLP(784, 10, 10).to(device)
            weak_learner = weak_learner.perform_training(train_loader, self._nn_num_epoch, self._nn_learning_rate)
            print("FINISHED TRAINING LEARNER", m)
            # Compute error of this weak learner
            incorrect = weak_learner.compute_incorrect_array(train_loader).to("cpu").numpy()
            # learner_err = sum([w[i] * self.I(y[i] != Gm(X_sample[i].reshape(1, -1))) for i in range(N)]) / sum(w)
            # learner_err = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
            # learner_err = np.average(incorrect, weights=w, axis=0)
            w = w.to("cpu").numpy()
            learner_err = np.dot(incorrect, w) / np.sum(w, axis=0)

            # If the learner's error is worse than randomly guessing, then we need to stop
            if learner_err >= 1 - 1 / num_classes:
                # TODO add for this
                print("EYY YOO big problem, for learner", m)
                break

            # Stop if we have gotten 0% error (i.e. 1e-10)
            if learner_err <= EPSILON:
                print("OMG 0 ERRORRRRR")
                break

            # Compute the alpha value (the "how much say"), also known as the learner weight
            alphaM = self._adaboost_learning_rate * (np.log((1 - learner_err) / learner_err) + np.log(num_classes - 1))

            if alphaM <= 0:
                print("EYY Yoo, early break, for learner", m)
                break

            # Update data distribution
            print("UPDAING WEIGHT ARRAY")
            w = self.update_weights_new(train_loader, incorrect, alphaM, w, self._nn_batch_size)
            print("DONE UPDATING")
            self._learner_array.append(weak_learner)
            self._learner_errors[m] = learner_err
            self._learner_weights[m] = alphaM
        return self._learner_errors, self._learner_weights

    def return_some_attribute(self):
        ...

    def perform_experiments(self):
        ...

    def ensembled_prediction(self, test_dataset):
        n_classes = 10
        classes_normal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        classes = classes_normal[:, np.newaxis]
        pred = sum((learner.predict_test_set(test_dataset) == classes).T * alpha
                   for learner, alpha in zip(self._learner_array, self._learner_weights))
        pred /= self._learner_weights.sum()

        final_predictions = classes_normal.take(np.argmax(pred, axis=1), axis=0)

        y_real = np.array([label for data, label, idx in test_dataset])
        print("Performance:", 100 * sum(y_real == final_predictions) / len(y_real))
        #print("Confusion Matrix:", "\n", CM(y_real, final_predictions))
        return final_predictions


# Trying out adaboost for CNN
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# cifar10_train = CustomDataset(dataset_name="CIFAR10", transform_to_apply=transform, train=True)
# adaboost_cnn = AdaBoost(2, [], [], cifar10_train, None, 1, 4, 0.001)
# adaboost_cnn.fit_ensemble()

# Trying out adaboost for mnist (mlp)
mnist_train = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=True)
mnist_test = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=False)

adaboost_mlp = AdaBoost(3, [], [], mnist_train, None, 5, 100, 0.001)
learner_err, learner_weight = adaboost_mlp.fit_ensemble()
print("learner errors", learner_err)
print("learner weights", learner_weight)
adaboost_mlp.ensembled_prediction(mnist_test)

