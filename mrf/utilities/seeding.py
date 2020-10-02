import random

import numpy as np
import torch


def do_seed(seed: int, with_cudnn: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if with_cudnn:
        # makes it perfectly deterministic but slower (without is already very good)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
