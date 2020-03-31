from catalyst.dl import registry, SupervisedRunner as Runner
from catalyst.contrib.models.cv import segmentation as m
from .experiment import Experiment
registry.MODELS.add_from_module(m)
