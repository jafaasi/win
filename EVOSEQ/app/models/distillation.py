from typing import Sequence, Optional, Union, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from .base import SequenceModel
from .ensemble import MetaEnsemble
from .dataset import SequenceDataset
from torch.utils.data import DataLoader
from ..features.encoding import one_hot

class KnowledgeDistiller:
    """
    Knowledge Distillation Engine:
    Distills high-capacity ensemble / teacher models into a compact student model
    using combined hard-label cross-entropy and soft-label KL divergence.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0, lr: float = 1e-3):
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr

    def distill(
        self,
        teacher: SequenceModel,
        student: SequenceModel,
        sequence: Sequence[int],
        epochs: int = 5,
        batch_size: int = 32
    ) -> SequenceModel:
        """Trains student model to match both ground truth outcomes and teacher probability distributions."""
        sequence = list(sequence)
        if len(sequence) < 20:
            return student
            
        # Collect soft teacher targets
        teacher_soft_targets = []
        for i in range(10, len(sequence)):
            ctx = sequence[:i]
            p_teach = teacher.predict_proba(ctx)
            teacher_soft_targets.append(p_teach)
            
        if hasattr(student, "net"):
            student.net.train()
            optimizer = torch.optim.Adam(student.net.parameters(), lr=self.lr)
            criterion_ce = nn.CrossEntropyLoss()
            criterion_kl = nn.KLDivLoss(reduction="batchmean")
            
            context_len = min(getattr(student, "context_length", 32), len(sequence)//2)
            inputs, hard_y, soft_y = [], [], []
            
            for idx, i in enumerate(range(10, len(sequence))):
                if i < context_len:
                    continue
                ctx = sequence[i - context_len:i]
                target = sequence[i]
                inputs.append(np.array([one_hot(int(d), student.input_size) for d in ctx], dtype=np.float32))
                hard_y.append(int(target))
                soft_y.append(teacher_soft_targets[idx])
                
            if not inputs:
                return student
                
            t_in = torch.tensor(np.array(inputs), dtype=torch.float32)
            t_hard = torch.tensor(hard_y, dtype=torch.long)
            t_soft = torch.tensor(np.array(soft_y), dtype=torch.float32)
            
            for _ in range(epochs):
                optimizer.zero_grad()
                logits = student.net(t_in)
                
                loss_hard = criterion_ce(logits, t_hard)
                log_probs_student = torch.log_softmax(logits / self.temperature, dim=-1)
                probs_teacher = torch.softmax(t_soft / self.temperature, dim=-1)
                loss_soft = criterion_kl(log_probs_student, probs_teacher) * (self.temperature ** 2)
                
                loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft
                loss.backward()
                nn.utils.clip_grad_norm_(student.net.parameters(), 1.0)
                optimizer.step()
        else:
            # Fallback for non-PyTorch models
            student.fit(sequence)
            
        return student
