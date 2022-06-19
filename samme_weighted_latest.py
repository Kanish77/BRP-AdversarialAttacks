import numpy as np
import torch
import torch.nn as nn
import torchvision
from advertorch.attacks import PGDAttack, GradientSignAttack
from advertorch.context import ctx_noparamgrad_and_eval
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from CNN import ConvNet
from MLP import MLP
from WCNN import WCNN2, WCNN3, WCNN4, WCCN5, CNNBaseLearner
from MyDataset import CustomDataset
from adversary_attack import AdversaryCode
import time
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

    def __init__(self, m, X, Y, training_data, base_learner, num_epochs, batch_size, nn_learning_rate, activation_fn,
                 learner_name, adv_train_algo, loss_fn, perturbation_radii):
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
        self._learner_name = learner_name

        # --------------------------------------------------------------------------------------------------------
        # The following are variables related to performing mini-experiments (including the independent variables)
        # --------------------------------------------------------------------------------------------------------
        self._nn_training_loss_function = loss_fn
        self._nn_activation_function = activation_fn
        self._nn_perturbation_radius = perturbation_radii
        self._nn_adversary_attack_algo = adv_train_algo

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

    def update_weights_random_sampled(self, train_loader, incorrect, alpha, w_prev, batch_size):
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
                w_new[i] = w_prev[i] * 1.0
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
        return torch.tensor(w_new, dtype=torch.float)

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

    def get_learner_from_name(self, activation_function):
        if self._learner_name == "WCNN2":
            return WCNN2().to(device)
        if self._learner_name == "WCNN3":
            return WCNN3(activation_function).to(device)
        elif self._learner_name == "WCNN4":
            return WCNN4().to(device)
        elif self._learner_name == "WCNN5":
            return WCCN5().to(device)
        else:
            raise ValueError("give valid name for learner")

    def fit_ensemble(self, adversarial_training, random_sampling=False):
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
            if random_sampling:
                if m >= 1:
                    train_loader = self.draw_random_sample(self._training_data, w, self._nn_batch_size)
                else:
                    train_loader = torch.utils.data.DataLoader(self._training_data, batch_size=self._nn_batch_size,
                                                               shuffle=True)
            else:
                train_loader = torch.utils.data.DataLoader(self._training_data, batch_size=self._nn_batch_size,
                                                           shuffle=True)

            # Train "weak" learner based on sample training data
            print("STARTED TRAINING LEARNER:", m)
            weak_learner = self.get_learner_from_name(self._nn_activation_function)

            # if we are doing adversarial training, we want to make the adversarial generation algo object
            if adversarial_training:
                # print("now doing adversarial training")
                adversary_algo = AdversaryCode(self._nn_perturbation_radius, weak_learner).get_adversary_method(self._nn_adversary_attack_algo)
                if self._nn_training_loss_function == "TRADES":
                    print("hey doing trades")
                    weak_learner = CNNBaseLearner.perform_trades_training(weak_learner, device, train_loader,
                                                                          self._nn_num_epoch, self._nn_learning_rate, w,
                                                                          adversary_algo)
                else:
                    weak_learner = CNNBaseLearner.perform_training(weak_learner, device, train_loader,
                                                                   self._nn_num_epoch,
                                                                   self._nn_learning_rate, w, adversarial_training=True,
                                                                   loss_fn=self._nn_training_loss_function,
                                                                   adversary_algo=adversary_algo,
                                                                   random_sampling=random_sampling)
                # Compute error of this weak learner based on the adversarial attacks being done to it
                incorrect = CNNBaseLearner.compute_incorrect_array_adversary_training(weak_learner, device,
                                                                                      train_loader,
                                                                                      adversary_algo).to("cpu").numpy()
            else:
                weak_learner = CNNBaseLearner.perform_training(weak_learner, device, train_loader, self._nn_num_epoch,
                                                               self._nn_learning_rate, w, adversarial_training=False,
                                                               loss_fn=self._nn_training_loss_function,
                                                               random_sampling=random_sampling)
                # Compute error of this weak learner
                incorrect = CNNBaseLearner.compute_incorrect_array(weak_learner, device, train_loader).to("cpu").numpy()

            print("FINISHED TRAINING LEARNER:", m, "AND COMPUTED IT'S ERRORS")

            # learner_err = sum([w[i] * self.I(y[i] != Gm(X_sample[i].reshape(1, -1))) for i in range(N)]) / sum(w)
            # learner_err = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
            # learner_err = np.average(incorrect, weights=w, axis=0)
            w = w.to("cpu").numpy()
            learner_err = np.dot(incorrect, w) / np.sum(w, axis=0)

            # If the learner's error is worse than randomly guessing, then we need to stop
            if learner_err >= 1 - 1 / num_classes:
                # TODO add for this
                print("EYY YOO big problem, for learner", m)
                # break

            # Stop if we have gotten 0% error (i.e. 1e-10)
            if learner_err <= EPSILON:
                print("OMG 0 ERRORRRRR")
                break

            # Compute the alpha value (the "how much say"), also known as the learner weight
            alphaM = self._adaboost_learning_rate * (np.log((1 - learner_err) / learner_err) + np.log(num_classes - 1))

            if alphaM <= 0:
                print("EYY Yoo, early break, for learner", m)
                # break

            # Update data distribution
            # print("UPDATING WEIGHTS")
            if random_sampling:
                w = self.update_weights_random_sampled(train_loader, incorrect, alphaM, w, self._nn_batch_size).to(
                    device)
            else:
                # print("using normal method, no random sampling")
                w = self.update_weights(train_loader, incorrect, alphaM, w, self._nn_batch_size).to(device)
            # print("DONE UPDATING")
            self._learner_array.append(weak_learner)
            self._learner_errors[m] = learner_err
            self._learner_weights[m] = alphaM
        # We want to return the best predictor from our learners
        best_learner = self._learner_array[np.argmax(self._learner_weights)]
        return self._learner_errors, self._learner_weights, best_learner

    def ensembled_prediction(self, test_dataset, adversarial_attack=False):
        n_classes = 10
        classes_normal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        classes = classes_normal[:, np.newaxis]
        pred = sum((CNNBaseLearner.predict_test_set(learner, device, test_dataset) == classes).T * alpha
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

    # def ensembled_adversary_robustness(self, test_dataset, attack_model, adversary_algo):
    #     classes_normal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #     classes = classes_normal[:, np.newaxis]
    #     pred = sum((CNNBaseLearner.predict_test_set_adversary(learner, device, test_dataset,
    #                                                           attack_model, adversary_algo) == classes).T * alpha
    #                for learner, alpha in zip(self._learner_array, self._learner_weights))
    #     pred /= self._learner_weights.sum()
    #
    #     final_predictions = classes_normal.take(np.argmax(pred, axis=1), axis=0)
    #
    #     y_real = np.array([label for data, label, idx in test_dataset])
    #
    #     print("Performance:", 100 * sum(y_real == final_predictions) / len(y_real), "\n")
    #     return final_predictions

    def test_adversary_robustness(self, test_dataset, adversary_method, target_model):
        # adversary_method = AdversaryCode("", target_model).get_adversary_method(adversary_algo)
        perturbed_images = []

        # For each image in the test dataset, we want to generate a perturbed image

        for image, true_label, idx in test_dataset:
            # image = image.reshape(-1, 28 * 28).to(device)
            image = image.reshape(1, 1, 28, 28).to(device)
            with ctx_noparamgrad_and_eval(target_model):
                perturbed_img = adversary_method.perturb(image, torch.tensor([true_label]).to(device))
            perturbed_images.append((perturbed_img, true_label, idx))

        final_predictions = self.ensembled_prediction(perturbed_images, True)

        # for i in range(3):
        #     plt.subplot(2, 3, i + 1)
        #     img = perturbed_images[i][0].reshape(28, 28)
        #     plt.imshow(img, cmap='gray')
        #     plt.title("perturbation images")
        # plt.show()
        return final_predictions

mnist_train = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=True)
mnist_test = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=False)
# # Settings for perturbation radii tests
# settings = []

# # Settings for activation function tests
# settings = [("ReLU", 7), ("ReLU", 1), ("HardTanH", 7), ("HardTanH", 1), ("Leaky_ReLU", 7), ("Leaky_ReLU", 1),
#             ("ELU", 7), ("ELU", 1), ("GeLU", 7), ("GeLU", 1)]

# # Settings for model size tests
# settings = [("WCNN3", True), ("WCNN3", False), ("WCNN4", True), ("WCNN4", False), ("WCNN5", True), ("WCNN5", False)]

# # Settings for adversarial algorithm tests
# settings = [("FGSM", 1), ("FGSM", 7), ("BIM", 1), ("BIM", 7), ("PGD", 1), ("PGD", 7), ("Lf-PGD", 1), ("Lf-PGD", 7)]

# # Settings for loss function tests
# settings = [("KL", 7), ("KL", 1), ("CE", 7), ("CE", 1), ("TRADES", 7), ("TRADES", 1)]

# For num_learners
# settings = {
#     "num_learners":   [1, 10],
#     "perturb_radii":  [0.3, 0.3],
#     "adversary_algo": ["PGD", "PGD"],
#     "model_size":     ["WCNN3", "WCNN3"],
#     "activation_fn":  ["ReLU", "ReLU"],
#     "loss_fn":        ["CE", "CE"],
# }

# # For perturbation_radii
# settings = {
#     "num_learners":   [7, 7],
#     "perturb_radii":  [0.03137, 0.30],
#     "adversary_algo": ["PGD", "PGD"],
#     "model_size":     ["WCNN3", "WCNN3"],
#     "activation_fn":  ["ReLU", "ReLU"],
#     "loss_fn":        ["CE", "CE"],
# }
#
# # For adversary_algo
# settings = {
#     "num_learners":   [7, 7],
#     "perturb_radii":  [0.3, 0.3],
#     "adversary_algo": ["FGSM", "PGD"],
#     "model_size":     ["WCNN3", "WCNN3"],
#     "activation_fn":  ["ReLU", "ReLU"],
#     "loss_fn":        ["CE", "CE"],
# }
#
# # For model type
# settings = {
#     "num_learners":   [1, 1, 1, 1, 1, 1],
#     "perturb_radii":  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
#     "adversary_algo": ["PGD", "PGD", "PGD", "PGD", "PGD", "PGD"],
#     "model_size":     ["WCNN4", "WCNN4", "WCNN4", "WCNN5", "WCNN5", "WCNN5"],
#     "activation_fn":  ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU"],
#     "loss_fn":        ["CE", "CE", "CE", "CE", "CE", "CE"],
# }
#
# # For activation function
# settings = {
#     "num_learners":   [7, 7],
#     "perturb_radii":  [0.3, 0.3],
#     "adversary_algo": ["PGD", "PGD"],
#     "model_size":     ["WCNN3", "WCNN3"],
#     "activation_fn":  ["ELU", "Leaky_ReLU"],
#     "loss_fn":        ["CE", "CE"],
# }
#
# # For loss function
# settings = {
#     "num_learners":   [7, 7, 7],
#     "perturb_radii":  [0.3, 0.3, 0.3],
#     "adversary_algo": ["PGD", "PGD", "PGD"],
#     "model_size":     ["WCNN3", "WCNN3", "WCNN3"],
#     "activation_fn":  ["ReLU", "ReLU", "ReLU"],
#     "loss_fn":        ["CE", "KL", "TRADES"],
# }

# Random experiments
# settings = {
#     "num_learners":   [1, 1, 1, 1, 7],
#     "perturb_radii":  [0.3, 0.3, 0.3, 0.3, 0.3],
#     "adversary_algo": ["MIA", "BIM", "BIM", "PGD", "PGD"],
#     "model_size":     ["WCNN3", "WCNN3", "WCNN3", "WCNN3", "WCNN3"],
#     "activation_fn":  ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU"],
#     "loss_fn":        ["CE", "CE", "CE", "KL", "KL"],
# }

# Single experiments
settings = {
    "num_learners":   [1, 4, 7, 1, 1, 1, 1, 1, 1],
    "perturb_radii":  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "adversary_algo": ["PGD", "PGD", "PGD", "PGD", "PGD", "PGD", "PGD", "PGD", "PGD"],
    "model_size":     ["WCNN3", "WCNN3", "WCNN3", "WCNN2", "WCNN4", "WCNN5", "WCNN3", "WCNN3", "WCNN3"],
    "activation_fn":  ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU"],
    "loss_fn":        ["CE", "CE", "CE", "CE", "CE", "CE", "CE", "KL", "TRADES"],
}

num_tests = len(settings["num_learners"])
for i in range(num_tests):
    num_learner = settings["num_learners"][i]
    perturb_radii = settings["perturb_radii"][i]
    adversary_algo = settings["adversary_algo"][i]
    model_size = settings["model_size"][i]
    activation_fn = settings["activation_fn"][i]
    loss_fn = settings["loss_fn"][i]
    print("Data for test with variable values: m = ", num_learner, ", e = ", perturb_radii, ", train adv_algo = ",
          adversary_algo, ", CNN type = ", model_size, ", activation_fn = ", activation_fn, ", loss_fn = ", loss_fn)
    print("------------------------------------------------------------------------------------------------------------------------------------------")
    adaboost_mlp = AdaBoost(num_learner, [], [], mnist_train, None, 3, 100, 0.002,
                            activation_fn=activation_fn, learner_name=model_size, adv_train_algo=adversary_algo,
                            loss_fn=loss_fn,  perturbation_radii=perturb_radii)
    start_time = time.time()
    learner_error, learner_weight, target = adaboost_mlp.fit_ensemble(adversarial_training=True,
                                                                      random_sampling=False)
    np.set_printoptions(precision=4)
    end_time = time.time()
    print("it took ", end_time-start_time, " seconds to train")
    # print("learner errors", learner_error)
    # print("learner weights", learner_weight)
    # adaboost_mlp.ensembled_prediction(mnist_test)

    # adversary_attack_PGD1 = PGDAttack(
    #     target, loss_fn=nn.CrossEntropyLoss(reduction="mean"),
    #     eps=0.15, nb_iter=20, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
    # adversary_attack_PGD2 = PGDAttack(
    #     target, loss_fn=nn.CrossEntropyLoss(reduction="mean"),
    #     eps=0.30, nb_iter=20, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
    # adversary_attack_FGSM = GradientSignAttack(
    #     target, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=0.156, targeted=False)

    # print("pgd attack1 now")
    # adaboost_mlp.test_adversary_robustness(mnist_test, adversary_attack_PGD1, target)
    # print("pgd attack2 now")
    # adaboost_mlp.test_adversary_robustness(mnist_test, adversary_attack_PGD2, target)
    # print("fgsm attack now")
    # adaboost_mlp.test_adversary_robustness(mnist_test, adversary_attack_FGSM, target)
    # print("--------------------------")
