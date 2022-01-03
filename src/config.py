import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PARAMS = {
    "data":
    {
        "input_dim": None, # done
        "target_idx": 3, # done
        "win_size": 50, # done
        "train_test_split": 0.7, # done
        "val_test_split": 0.2, # done
    },

    "models":
    {
        "hidden_dim": 60, # done
        "kernel_size": 3, # done
        "bias": True, # done
        "num_layers": 1, # done
        "dropout": 0.15, # done
        "n_att_heads": 8, # done
        "num_steps_to_predict": 10,
        "C": 0.01, # done
        "epsilon": 0.01 # done
    },

    "optimize":
    {
        "lr": 1e-3, # done
        "num_epochs": 10, # done
        "optimizer": optim.Adam, # done
        "criterion": nn.MSELoss(), # done
        "verbose": False # done
    },
}