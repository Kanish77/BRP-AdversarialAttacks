import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from advertorch.attacks import LinfBasicIterativeAttack, GradientSignAttack
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack, LinfPGDAttack, MomentumIterativeAttack


class AdversaryCode(object):

    def __init__(self, radii, defense_obj):
        self.radii = radii
        self.def_model = defense_obj


    """
    For a given attack type, this method returns the associating adversary method that can perturb an image
    """
    def get_adversary_method(self, attack_name):
        if attack_name == "FGSM":
            print("Doing adversarial training with FGSM")
            adversary = GradientSignAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=self.radii, targeted=False)
        elif attack_name == "BIM":
            print("hey training with BIM")
            adversary = LinfBasicIterativeAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=self.radii,
                nb_iter=20, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
        elif attack_name == "CW":
            adversary = CarliniWagnerL2Attack(
                self.def_model, num_classes=10, learning_rate=0.45, binary_search_steps=10,
                max_iterations=12, targeted=False)
        elif attack_name == "PGD":
            print("hey training with PGD")
            adversary = PGDAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="mean"),
                eps=self.radii, nb_iter=20, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
        elif attack_name == "Lf-PGD":
            print("hey training with lf-pgd")
            adversary = LinfPGDAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=self.radii,
                nb_iter=20, eps_iter=0.03, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
        elif attack_name == "MIA":
            print("hey training with MIA")
            adversary = MomentumIterativeAttack(
                self.def_model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=self.radii, nb_iter=20,
                eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
        else:
            # Using clean data samples
            adversary = None

        return adversary
