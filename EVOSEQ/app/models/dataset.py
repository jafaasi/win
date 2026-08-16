import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Union, Sequence
from ..features.encoding import one_hot

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for causal temporal sequence training.
    Maps (X_{t-L}, ..., X_{t-1}) -> y_t without future information leakage.
    """

    def __init__(
        self,
        features: Union[np.ndarray, Sequence],
        targets: Union[np.ndarray, Sequence],
        context_length: int = 128,
        input_size: int = 10,
    ):
        raw_feat = np.asarray(features)
        if raw_feat.ndim == 1:
            # Convert 1D integer digit stream to one-hot: [N, 10]
            self.features = np.array([one_hot(int(d), input_size) for d in raw_feat], dtype=np.float32)
        else:
            self.features = raw_feat.astype(np.float32)
            
        self.targets = np.asarray(targets, dtype=np.int64)
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.features) - self.context_length)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = index + self.context_length
        X = self.features[start:end]
        y = self.targets[end]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
