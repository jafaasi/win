from typing import Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiHorizonDataset(Dataset):
    """
    Constructs multi-horizon temporal training slices:
    Given: X[t-context:t]
    Targets: y = [x[t+1], x[t+2], ..., x[t+H]]
    """

    def __init__(
        self,
        features: Sequence[np.ndarray],
        digits: Sequence[int],
        context_length: int = 128,
        horizons: int = 3,
    ):
        self.features = np.asarray(features, dtype=np.float32)
        self.digits = np.asarray(digits, dtype=np.int64)
        self.context = context_length
        self.horizons = horizons
        self.length = max(0, len(self.features) - context_length - horizons + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = index + self.context
        X = self.features[index:end]
        y = self.digits[end:end + self.horizons]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
