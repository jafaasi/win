from typing import Sequence, Optional, Union, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from ..base import SequenceModel, ModelMetadata
from ..neural.trainer import get_device, train_epoch, calculate_loss
from ..neural.dataset import MultiHorizonDataset
from ...features.vector import encode_observation
from ...core.mapping import map_digit
from torch.utils.data import DataLoader

class SSMAdapter(SequenceModel):
    """
    Universal adapter integrating State-Space Models (Mamba, Mamba-2, S4, S4D)
    into the EVOSEQ SequenceModel polymorphism hierarchy.
    """

    def __init__(
        self,
        model: nn.Module,
        context_length: int = 128,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 32,
        name: str = "SSM-Adapter",
        version: str = "ssm-v1"
    ):
        self.model = model
        self.context_length = context_length
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = get_device()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.metadata = ModelMetadata(
            name=name,
            version=version,
            parameters={"context_length": context_length, "lr": lr, "epochs": epochs}
        )

    def _sequence_to_features(self, sequence: Sequence[int]) -> np.ndarray:
        features = []
        for d in sequence:
            m = map_digit(int(d))
            features.append(encode_observation(m["digit"], m["size"], m["color"], m["parity"]))
        return np.asarray(features, dtype=np.float32)

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "SSMAdapter":
        sequence = list(X)
        if len(sequence) <= (self.context_length + 3):
            return self
            
        features = self._sequence_to_features(sequence)
        dataset = MultiHorizonDataset(features, sequence, context_length=self.context_length, horizons=3)
        if len(dataset) == 0:
            return self
            
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            train_epoch(self.model, loader, self.optimizer, self.device)
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "SSMAdapter":
        return self.fit(X, y)

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Returns 10-class probability distribution for Horizon 1 given context sequence."""
        sequence = list(X)
        if len(sequence) < self.context_length:
            pad = [sequence[0] if sequence else 0] * (self.context_length - len(sequence))
            sequence = pad + sequence
            
        context_seq = sequence[-self.context_length:]
        features = self._sequence_to_features(context_seq) # [context_length, 17]
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, L, 17]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            h1_logits = outputs[0][0] # [10]
            probs = torch.softmax(h1_logits, dim=-1).cpu().numpy().astype(np.float64)
            return probs

    def predict_multi_horizon(self, X: Union[np.ndarray, list]) -> List[np.ndarray]:
        """Returns probability distributions for [H1, H2, H3]."""
        sequence = list(X)
        if len(sequence) < self.context_length:
            pad = [sequence[0] if sequence else 0] * (self.context_length - len(sequence))
            sequence = pad + sequence
            
        context_seq = sequence[-self.context_length:]
        features = self._sequence_to_features(context_seq)
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            return [torch.softmax(out[0], dim=-1).cpu().numpy().astype(np.float64) for out in outputs]

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "metadata": self.metadata,
            "context_length": self.context_length
        }, path if path.endswith(".pt") else f"{path}.pt")

    @classmethod
    def load(cls, path: str, model_instance: Optional[nn.Module] = None) -> "SSMAdapter":
        data = torch.load(path if path.endswith(".pt") else f"{path}.pt", map_location="cpu", weights_only=False)
        if model_instance is None:
            from .s4_model import S4DSequenceModel
            model_instance = S4DSequenceModel()
        model_instance.load_state_dict(data["model_state"])
        adapter = cls(model=model_instance, context_length=data.get("context_length", 128))
        adapter.metadata = data.get("metadata", adapter.metadata)
        return adapter
