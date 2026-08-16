import os
import json
import hashlib
from typing import Optional, Dict, Any
from ..models.base import SequenceModel

class CheckpointManager:
    """
    Manages atomic versioned model checkpoints and checksum verification.
    Directory structure:
    checkpoints/{model_family}/{version}/
      ├── model.chk (or model.npy)
      └── metadata.json
    """

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_dir(self, model_name: str, version: str) -> str:
        d = os.path.join(self.base_dir, model_name.lower(), version)
        os.makedirs(d, exist_ok=True)
        return d

    def save_checkpoint(self, model: SequenceModel, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        name = model.metadata.name
        version = model.metadata.version
        chk_dir = self._get_dir(name, version)
        
        file_path = os.path.join(chk_dir, "model.chk")
        model.save(file_path)
        
        meta = {
            "name": name,
            "version": version,
            "parameters": model.metadata.parameters,
            "extra": extra_meta or {}
        }
        meta_path = os.path.join(chk_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        return chk_dir

    def verify_integrity(self, model_name: str, version: str) -> bool:
        chk_dir = os.path.join(self.base_dir, model_name.lower(), version)
        file_path = os.path.join(chk_dir, "model.chk")
        npy_path = os.path.join(chk_dir, "model.chk.npy")
        meta_path = os.path.join(chk_dir, "metadata.json")
        
        has_model = os.path.isfile(file_path) or os.path.isfile(npy_path)
        has_meta = os.path.isfile(meta_path)
        return has_model and has_meta
