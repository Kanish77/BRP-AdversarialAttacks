import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from advertorch.attacks import LinfBasicIterativeAttack, GradientSignAttack
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack, LinfPGDAttack


class AdversaryCode(object):

    def __init__(self, args, defense_obj=None):
        self.args = args
        self.adversary = None
        self.train_loader = None
        self.test_loader = None
        self.def_model = defense_obj


    """
    For a given attack type, this method returns the associating adversary method that can perturb an image
    """
    def get_adversary_method(self, attack_name):
        if attack_name == "FGSM":
            adversary = GradientSignAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=0.15, targeted=False)
        elif attack_name == "BIM":
            adversary = LinfBasicIterativeAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
                nb_iter=200, eps_iter=0.02, clip_min=0.0, clip_max=1.0, targeted=False)
        elif attack_name == "CW":
            adversary = CarliniWagnerL2Attack(
                self.def_model, num_classes=10, learning_rate=0.45, binary_search_steps=10,
                max_iterations=12, targeted=False)
        elif attack_name == "PGD":
            adversary = PGDAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="mean"),
                eps=0.078, nb_iter=6, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
        elif attack_name == "Lf-PGD":
            adversary = LinfPGDAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
                nb_iter=6, eps_iter=0.03, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
        else:
            # Using clean data samples
            adversary = None

        return adversary

    """
    This method generates an adversarial example for adversarial training.
    """
    def generate_adversarial_example(self, adversary, image, label):
        return adversary.perturb(image, label)


    """
    This method will attack a given model ie. used for white-box attack testing
    """
    def attack_model(self):
        ...
