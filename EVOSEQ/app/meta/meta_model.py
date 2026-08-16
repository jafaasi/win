from typing import Sequence, Dict, Any, Tuple, Optional, List
import numpy as np
from .types import EnvironmentState, ModelDescriptor
from ..database import SessionLocal
from ..schemas import MetaExperimentRecord

class MetaModel:
    """
    Pure NumPy Bayesian Ridge Meta-Learner:
    Predicts expected validation score and analytical Bayesian posterior uncertainty (mu, sigma)
    using closed-form Gaussian Process / Ridge posterior covariance:
    w = (X^T X + lambda * I)^(-1) X^T y
    Sigma = (X^T X + lambda * I)^(-1)
    mu(x) = x^T w
    sigma(x) = sqrt(sigma_noise^2 + x^T Sigma x)
    UCB(x) = mu(x) + kappa * sigma(x)
    """

    def __init__(self, lambda_reg: float = 1.0, noise_variance: float = 0.04):
        self.lambda_reg = lambda_reg
        self.noise_variance = noise_variance
        self.weights: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self._is_fitted = False

    def _build_feature_vector(self, env: EnvironmentState, desc: ModelDescriptor) -> np.ndarray:
        raw = np.concatenate([env.to_vector(), desc.to_vector()])
        # Append bias term
        return np.concatenate([[1.0], raw])

    def fit_from_database(self, train_ratio: float = 0.8) -> bool:
        with SessionLocal() as session:
            records = session.query(MetaExperimentRecord).order_by(MetaExperimentRecord.id.asc()).all()
            if len(records) < 5:
                return False
                
            X_list, y_list = [], []
            for r in records:
                env = EnvironmentState(**r.environment)
                desc = ModelDescriptor(**r.model_descriptor)
                score = (r.null_advantage or 0.0) - ((r.brier_score or 0.09) * 5.0) - ((r.calibration_error or 0.02) * 2.0)
                X_list.append(self._build_feature_vector(env, desc))
                y_list.append(score)
                
            X = np.array(X_list, dtype=np.float64)
            y = np.array(y_list, dtype=np.float64)
            
            # Strict chronological time split
            split_idx = int(len(X) * train_ratio)
            X_train, y_train = X[:split_idx], y[:split_idx]
            
            if len(X_train) < 3:
                X_train, y_train = X, y
                
            D = X_train.shape[1]
            A = X_train.T @ X_train + self.lambda_reg * np.eye(D)
            self.cov_matrix = np.linalg.inv(A)
            self.weights = self.cov_matrix @ X_train.T @ y_train
            self._is_fitted = True
            return True

    def predict_ucb(self, env: EnvironmentState, desc: ModelDescriptor, kappa: float = 1.96) -> Tuple[float, float, float]:
        """
        Computes Bayesian Upper Confidence Bound:
        UCB = mu + kappa * sigma.
        """
        x = self._build_feature_vector(env, desc)
        if not self._is_fitted or self.weights is None or self.cov_matrix is None:
            # Prior defaults if insufficient meta-history
            return -0.40, 0.15, round(-0.40 + (kappa * 0.15), 4)
            
        mu = float(x @ self.weights)
        epistemic_var = float(x.T @ self.cov_matrix @ x)
        sigma = float(np.sqrt(max(1e-6, self.noise_variance + epistemic_var)))
        ucb = float(mu + kappa * sigma)
        return round(mu, 4), round(sigma, 4), round(ucb, 4)
