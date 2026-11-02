"""Read tensors or slices of them straight to a GPU; nothing stays in host RAM."""
import glob
import os

from safetensors import safe_open

P = "model.language_model."


class Checkpoint:
    def __init__(self, snapshot):
        self.f = safe_open(glob.glob(os.path.join(snapshot, "*.safetensors"))[0], "pt")

    def full(self, name, device):
        return self.f.get_tensor(P + name).to(device, non_blocking=True)

    def rows(self, name, start, end, device):
        """weight[start:end, :]"""
        return self.f.get_slice(P + name)[start:end].to(device, non_blocking=True)

    def cols(self, name, start, end, device):
        """weight[:, start:end]"""
        return self.f.get_slice(P + name)[:, start:end].to(device, non_blocking=True)
