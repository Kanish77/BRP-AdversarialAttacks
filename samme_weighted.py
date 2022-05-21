import numpy as np
import torch
import torch.nn as nn
import torchvision
from advertorch.context import ctx_noparamgrad_and_eval
from sklearn.metrics import confusion_matrix as CM
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from CNN import ConvNet
from MLP import MLP
from MyDataset import CustomDataset
from adversary_attack import AdversaryCode

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
        self._nn_adversary_attack_algo = "FGSM"
        self._nn_optimiser = "SGD"

    def update_weights(self, train_loader, incorrect, alpha, w_prev, batch_size):
        w_new = torch.zeros(len(w_prev))
        for images, labels, index in train_loader:
            for i in range(batch_size):
                idx = index[i].item()
                w_update_value = w_prev[idx] * np.exp(alpha * incorrect[idx])
                w_new[idx] = w_update_value
        # check for non-negative total sum or ∞ sum
        w_sum = sum(w_new).item()
        if w_sum <= 0:
            print("EYY Yoo, weights negative, for learner")
            raise ValueError("weights are negative")
        if not np.isfinite(w_sum):
            raise ValueError("Sample weights have reached infinite values, causing overflow. "
                             "Iterations stopped. Try lowering the learning rate.")
        # normalise
        w_new = w_new / w_sum
        return w_new

    def fit_ensemble(self, adversarial_training):
        # We need to first do some initializations for the first ensemble.
        N = len(self._training_data)

        # The weight array stores the weight of training data x_i at index i (thus w[0] corresponds to weight of
        # the very first training data
        w = torch.tensor([1 / N for i in range(N)]).to(device)
        w = w / torch.sum(w)
        y = self._Y

        # self._classes = np.array(sorted(list(set(y))))
        # num_classes = len(self._classes)
        # self._num_classes = num_classes
        num_classes = 10
        self._num_classes = num_classes

        for m in range(self._number_learners):
            train_loader = torch.utils.data.DataLoader(self._training_data, batch_size=self._nn_batch_size,
                                                       shuffle=True)

            # Train "weak" learner based on sample training data
            print("STARTED TRAINING LEARNER:", m)
            weak_learner = MLP(784, 10, 10, device).to(device)
            # if we are doing adversarial training, we want to make the adversarial generation algo object
            if adversarial_training:
                adversary_algo = AdversaryCode("", weak_learner).get_adversary_method(self._nn_adversary_attack_algo)
                weak_learner = weak_learner.perform_training(train_loader, self._nn_num_epoch, self._nn_learning_rate,
                                                             w, adversarial_training=True, adversary_algo=adversary_algo)
            else:
                weak_learner = weak_learner.perform_training(train_loader, self._nn_num_epoch, self._nn_learning_rate,
                                                             w, adversarial_training=False)
            print("FINISHED TRAINING LEARNER:", m)

            # Compute error of this weak learner
            if adversarial_training:
                adversary_algo = AdversaryCode("", weak_learner).get_adversary_method(self._nn_adversary_attack_algo)
                incorrect = weak_learner.compute_incorrect_array_adversary_training(train_loader, adversary_algo).to("cpu").numpy()
            else:
                incorrect = weak_learner.compute_incorrect_array(train_loader).to("cpu").numpy()
            # learner_err = sum([w[i] * self.I(y[i] != Gm(X_sample[i].reshape(1, -1))) for i in range(N)]) / sum(w)
            # learner_err = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
            # learner_err = np.average(incorrect, weights=w, axis=0)
            w = w.to("cpu").numpy()
            learner_err = np.dot(incorrect, w) / np.sum(w, axis=0)

            # If the learner's error is worse than randomly guessing, then we need to stop
            # if learner_err >= 1 - 1 / num_classes:
            #     # TODO add for this
            #     print("EYY YOO big problem, for learner", m)
            #     break

            # Stop if we have gotten 0% error (i.e. 1e-10)
            if learner_err <= EPSILON:
                print("OMG 0 ERRORRRRR")
                break

            # Compute the alpha value (the "how much say"), also known as the learner weight
            alphaM = self._adaboost_learning_rate * (np.log((1 - learner_err) / learner_err) + np.log(num_classes - 1))

            # if alphaM <= 0:
            #     print("EYY Yoo, early break, for learner", m)
            #     break

            # Update data distribution
            print("UPDATING WEIGHTS")
            w = self.update_weights(train_loader, incorrect, alphaM, w, self._nn_batch_size)
            print("DONE UPDATING")
            self._learner_array.append(weak_learner)
            self._learner_errors[m] = learner_err
            self._learner_weights[m] = alphaM
        # We want to return the best predictor from our learners
        best_learner = self._learner_array[np.argmax(self._learner_weights)]
        return self._learner_errors, self._learner_weights, best_learner

    def return_some_attribute(self):
        ...

    def perform_experiments(self):
        ...

    def ensembled_prediction(self, test_dataset, adversarial_attack=False):
        n_classes = 10
        classes_normal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        classes = classes_normal[:, np.newaxis]
        pred = sum((learner.predict_test_set(test_dataset) == classes).T * alpha
                   for learner, alpha in zip(self._learner_array, self._learner_weights))
        pred /= self._learner_weights.sum()

        final_predictions = classes_normal.take(np.argmax(pred, axis=1), axis=0)

        y_real = np.array([label for data, label, idx in test_dataset])

        if adversarial_attack:
            print("Adversarial Attack Information")
        else:
            print("Normal Test Information")
        print("Performance:", 100 * sum(y_real == final_predictions) / len(y_real), "\n")
        # print("Confusion Matrix:", "\n", CM(y_real, final_predictions))
        return final_predictions

    def test_adversary_robustness(self, test_dataset, adversary_algo, target_model):
        adversary_method = AdversaryCode("", target_model).get_adversary_method(adversary_algo)
        perturbed_images = []

        # For each image in the test dataset, we want to generate a perturbed image

        for image, true_label, idx in test_dataset:
            image = image.reshape(-1, 28 * 28)
            with ctx_noparamgrad_and_eval(target_model):
                perturbed_img = adversary_method.perturb(image, torch.tensor([true_label]))
            perturbed_images.append((perturbed_img, true_label, idx))

        final_predictions = self.ensembled_prediction(perturbed_images, True)

        for i in range(3):
            plt.subplot(2, 3, i + 1)
            img = perturbed_images[i][0].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title("perturbation images")
        plt.show()
        return final_predictions


# # Trying out adaboost for CNN
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# cifar10_train = CustomDataset(dataset_name="CIFAR10", transform_to_apply=transform, train=True)
# cifar10_test = CustomDataset(dataset_name="CIFAR10", transform_to_apply=transform, train=False)
# adaboost_cnn = AdaBoost(2, [], [], cifar10_train, None, 1, 250, 0.001)
# learner_err, learner_weight = adaboost_cnn.fit_ensemble()
# print("learner errors", learner_err)
# print("learner weights", learner_weight)
# adaboost_cnn.ensembled_prediction(cifar10_test)

# Trying out adaboost for mnist (mlp)
mnist_train = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=True)
mnist_test = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=False)

adaboost_mlp = AdaBoost(5, [], [], mnist_train, None, 5, 100, 0.001)
learner_error, learner_weight, target = adaboost_mlp.fit_ensemble(adversarial_training=True)
print("learner errors", learner_error)
print("learner weights", learner_weight)
adaboost_mlp.ensembled_prediction(mnist_test)
adaboost_mlp.test_adversary_robustness(mnist_test, "FGSM", target)

for i in range(3):
    plt.subplot(2, 3, i + 1)
    plt.imshow(mnist_test[i][0][0], cmap='gray')
    plt.title("real images")
plt.show()
