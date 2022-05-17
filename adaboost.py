import numpy as np
import torch
import torch.nn as nn
import torchvision
import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix as CM
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


from numpy.core.umath_tests import inner1d


class AdaBoost(object):
    """
    Adaboost constructor.
    @param m - number of learners Adaboost will use in ensembling
    @param X - training data X, of length n
    @param y - training labels corresponding to data, of length n
    @param independent_variable - if performing experiments, which variable is being changed
    @param independent_variable_values - if performing experiments, pass an array of all the potential values that you want
        to try
    """

    def __init__(self, m, X, y, independent_variable="ALL", independent_variable_values=[]):
        # Defining the number of learners
        self._number_learners = m
        # Defining the learning rate of the learners. Weight applied to each classifier at each boosting iteration.
        # A higher learning rate increases the contribution of each classifier. There is a trade-off between the
        # `learning_rate` and `n_estimators` parameters.
        self._learning_rate = 1
        # Making an array for the learners (these can be pytorch CNN objects or whatever machine learning classifier)
        self._learner_array = []
        # An array that stores the "amount of say" of each learner i.e its weight in the final prediction
        self._learner_weights = np.zeros(self._number_learners)
        # An array that stores the error of each learner
        self._learner_errors = np.ones(self._number_learners)

        # Data related to training data
        self._original_X = X
        self._y = y

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
    Returns a tuple, (X_sample_entry, idx), where X_sample_entry is an entry of the sampled training data
    and the idx value corresponds to the index of the original training data (thus, if a single training data is sampled
    multiple times, then their indexes will be the same, and we can then use this to update the corresponding training weight)
    
    In other words, if indicies[5] = 46, that means that the 5th sampled training example is equal to the 46th original 
    training example
    
    Note: it is assumed that the inputed training weights are NORMALISED.
    """

    def draw_random_sample(self, X_original, normal_training_weights):
        # normal_training_weights = np.array([0.071475, 0.071475, 0.166559, 0.071475, 0.071475, 0.071475, 0.166559, 0.071475, 0.166559, 0.071475])
        # normal_training_weights = normal_training_weights / normal_training_weights.sum()
        # Here we compute the cumulative distribution values i.e for each training example, its upper and lower bound
        cumsum_upper = np.cumsum(normal_training_weights)
        cumsum_lower = cumsum_upper - normal_training_weights

        # Based on this distribution, we perform random sampling with replacement
        random_sample = []
        indices = []
        for i in range(len(normal_training_weights)):
            p = np.random.random()
            for k, data in enumerate(X_original):
                if cumsum_upper[k] > p and p > cumsum_lower[k]:
                    random_sample.append(data)
                    indices.append(k)
                    break
        return random_sample, indices

    def update_weights(self, X_sample, y, G, alpha, w_prev, indices):
        N = len(y)
        w_updates = [[] for i in range(N)]

        for i in range(N):
            w_update_value = w_prev[i] * np.exp(alpha * self.I(y[i] != G.predict(X_sample[i].reshape(1, -1))))
            original_training_index = indices[i]
            w_updates[original_training_index].append(w_update_value)

        # each weight is averaged
        w_new = np.zeros(N)
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
        return w_new

    def fit_ensemble(self):
        # We need to first do some initializations for the first ensemble. This includes preparing data
        # setting initial weights
        self._original_X = np.float64(self._original_X)
        N = len(self._y)
        # The weight array stores the weight of training data x_i at index i (thus w[0] corresponds to weight of
        # original_X[0])
        w = np.array([1 / N for i in range(N)])
        y = self._y

        self._classes = np.array(sorted(list(set(y))))
        num_classes = len(self._classes)
        self._num_classes = num_classes

        print_once = True

        for m in range(self._number_learners):
            # Sample data based on weight distribution (we don't if m == 0 i.e. we are training the first learner)
            if m > 1:
                X_sample, idx = self.draw_random_sample(self._original_X, w)
            else:
                X_sample = self._original_X
                idx = [i for i in range(N)]

            # Train "weak" learner based on sample training data
            # TODO, when i make this non-hardcoded, i need to add randomness with seeds for the base classifier
            # TODO / for example, see scikit impl or Adaboost-CNN code
            Gm = DecisionTreeClassifier(max_depth=3).fit(X_sample, y, sample_weight=w)

            # Compute error of this weak learner
            y_pred = Gm.predict(X_sample)
            incorrect = y_pred != y

            if print_once:
                #print("y_original", y)
                #print("y_pred", y_pred)
                #print("incorrect", incorrect)
                print_once = False

            # learner_err = sum([w[i] * self.I(y[i] != Gm(X_sample[i].reshape(1, -1))) for i in range(N)]) / sum(w)
            # learner_err = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
            learner_err = np.average(incorrect, weights=w, axis=0)

            # If the learner's error is worse than randomly guessing, then we need to stop
            if learner_err >= 1 - 1 / num_classes:
                # TODO add shite for this
                print("EYY YOO big problem, for learner", m)
                break

            # Stop if we have gotten 0% error.
            if learner_err == 0:
                break

            # Compute the alpha value (the "how much say"), also known as the learner weight
            alphaM = self._learning_rate * (np.log((1 - learner_err) / learner_err) + np.log(num_classes - 1))

            if alphaM <= 0:
                print("EYY Yoo, early break, for learner", m)
                break

            # Update data distribution
            w = self.update_weights(X_sample, y, Gm, alphaM, w, idx)

            self._learner_array.append(Gm)
            self._learner_errors[m] = learner_err
            self._learner_weights[m] = alphaM

    def make_prediction(self, X):
        y = 0
        for m in range(self._number_learners):
            Gm = self._learner_array[m]
            AlphaM = self._learner_weights[m]
            y += AlphaM * Gm.predict(X)
        signA = np.vectorize(self.sign)
        y = np.where(signA(y) == -1, -1, 1)
        return y

    def return_some_attribute(self):
        ...

    def perform_experiments(self):
        ...

    def predict(self, X):
        print_once = True
        n_classes = self._num_classes
        classes = self._classes[:, np.newaxis]
        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(self._learner_array, self._learner_weights))
        pred /= self._learner_weights.sum()

        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self._classes.take(pred > 0, axis=0)

        if print_once:
            estimator = self._learner_array[0]
            print("n_classes: ", n_classes)
            print("classes with odd np shite", classes)
            print("classes normal", self._classes)
            print("pred normalised", pred)
            res = self._classes.take(np.argmax(pred, axis=1), axis=0)
            print("final result", res)
            print("estimator.predict(X)", estimator.predict(X))
            res2 = estimator.predict(X) == classes
            print("estimator.predict(X)==classes", res2)
            print("but now with T", res2.T)

        return self._classes.take(np.argmax(pred, axis=1), axis=0)


#x, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

digits = datasets.load_digits()
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.8, shuffle=False)

adaboost = AdaBoost(m=5, X=X_train, y=y_train)
adaboost.fit_ensemble()
y_pred_orig = adaboost.predict(X_test)
print("Performance:", 100 * sum(y_pred_orig == y_test) / len(y_test))
print("Confusion Matrix:", "\n", CM(y_test, y_pred_orig))




# clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME")
# clf.fit(X_train, y_train)
# y_pred2 = clf.predict(X_test)
# print("Performance:", 100 * sum(y_pred2 == y_test) / len(y_test))
# print("Confusion Matrix:\n", CM(y_test, y_pred2))

# Code from https://github.com/Samarendra109/ML-Models/blob/master/ensemble/AdaBoostClassifier.py
# def indexToVector(y, k, labelDict):
#     y_new = []
#     for yi in y:
#         i = labelDict[yi]
#         v = np.ones(k) * (-1 / (k - 1))
#         v[i] = 1
#         y_new.append(v)
#     return np.array(y_new)
#
# def indexToLabel(i, clf):
#     return clf[i]
#
# dict = {'1': 0, '2': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6}
# y_pred = sum(
#     Bm * indexToVector(gm_pred, 7, dict) for Bm, gm_pred in [(0.2, ["C", "A"]), (0.3, ["C", "B"]), (0.9, ["B", "A"])])
# print(y_pred)
# iTL = np.vectorize(indexToLabel)
#
# classes = np.array(['1' '2' 'A' 'B' 'C' 'D' 'E'])
# res = iTL(np.argmax(y_pred, axis=1), classes)
# print(res)
