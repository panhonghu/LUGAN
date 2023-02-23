from .preparation import DatasetSimple, DatasetDetections
from .gait import (CasiaBPose,)
from .sampler import RandomIdentitySampler


def dataset_factory(name):
    if name == "casia-b":
        return CasiaBPose
    raise ValueError()
