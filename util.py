import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # disable cuDNN benchmarking and cuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
