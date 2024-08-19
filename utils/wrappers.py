import torch
import numpy as np
from autoencoder.utils.loss_function import symexp


class CAEWrapper(torch.nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.cae = model
        self.mode = mode
        if not (mode in ['encode', 'decode']):
            raise RuntimeError(f"Mode must be either encode or decode. Got {mode}.")

    def forward(self, x):
        batch_size = x.shape[:-1]
        observation_size = x.shape[-1]
        # Resize to correct shape for CAE
        if self.mode == 'encode':
            x = x.view(int(np.prod(batch_size)), 1, observation_size)
        elif self.mode == 'decode':
            x = x.view(int(np.prod(batch_size)), observation_size)
        # Map through model
        x = self.cae(x)
        # x = self.cae.symp_forward(x)
        if self.mode == 'decode':
            # The decoder returns the output in symlog space.
            # We map through symexp to map back to physical space.
            x = symexp(x)
        """  # retire
        if isinstance(x, tuple):
            # If we get both encoded and decoded, the model is the full CAE
            # In this case, we want to keep only decoded and map through symexp
            x = x[1]
            x = symexp(x)
        """
        # Reshape to correct shape for RL training
        x = x.view(*batch_size, x.shape[-1])
        return x

