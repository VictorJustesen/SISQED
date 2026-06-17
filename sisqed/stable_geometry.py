from dataclasses import dataclass

import numpy as np

from isqed.core import Intervention, ModelUnit
from isqed.geometry import DISCOSolver


@dataclass
class StableAuditResult:
    weights: np.ndarray
    naive_score: float
    stable_magnitude: float
    stability_variance: float
    stable_score: float
    mean_residuals: np.ndarray
    residual_samples: np.ndarray


class StableDISCOSolver:

    @staticmethod
    def _query_batch(model: ModelUnit, X_batch: np.ndarray, seeds) -> np.ndarray:
        if hasattr(model, "predict_batch_with_seeds"):
            values = model.predict_batch_with_seeds(X_batch, seeds)
        elif hasattr(model, "predict_with_seed"):
            values = [model.predict_with_seed(x_i, seed) for x_i, seed in zip(X_batch, seeds)]
        else:
            values = model._forward(X_batch)

        if getattr(model, "scalarizer", None) is not None:
            values = [model.scalarizer(v) for v in np.asarray(values)]

        return np.asarray(values, dtype=float).reshape(len(X_batch))
        
    @staticmethod
    def _collect_repeated_responses(
        target: ModelUnit,
        peers: list[ModelUnit],
        X,
        thetas,
        intervention: Intervention,
        repeats: int,
        seed_offset: int = 0,
    ):
        """collect repated respnoses on diffferent queires with different seeds"""
        x_list = [np.asarray(x, dtype=float) for x in X]

        if isinstance(thetas, (int, float)):
            theta_list = [float(thetas)] * len(x_list)
        else:
            theta_list = list(thetas)

        n_pairs = len(x_list)
        if len(theta_list) != n_pairs:
            raise ValueError("Length mismatch between `X` and `thetas`.")

        y_target = np.zeros((n_pairs, repeats), dtype=float)
        y_peers = np.zeros((n_pairs, repeats, len(peers)), dtype=float)

        for s in range(repeats):
            seeds = [seed_offset + i * repeats + s for i in range(n_pairs)]
            x_perturbed = np.asarray(
                [
                    intervention.apply(x_i, theta_i, seed)
                    for x_i, theta_i, seed in zip(x_list, theta_list, seeds)
                ],
                dtype=float,
            )

            y_target[:, s] = StableDISCOSolver._query_batch(target, x_perturbed, seeds)

            for j, peer in enumerate(peers):
                y_peers[:, s, j] = StableDISCOSolver._query_batch(peer, x_perturbed, seeds)

        return y_target, y_peers

    @staticmethod
    def solve_stable_weights(
        target_means: np.ndarray,
        peer_means: np.ndarray,
    ):
        """build the synthetic clone"""
        target_means = np.asarray(target_means, dtype=float).reshape(-1)
        peer_means = np.asarray(peer_means, dtype=float)

        n_pairs, n_peers = peer_means.shape
        if len(target_means) != n_pairs:
            raise ValueError(
                f"Shape mismatch: target {target_means.shape} vs peers {peer_means.shape}"
            )

        _, weights = DISCOSolver.solve_weights_and_distance(
            target_vec=target_means,
            peer_matrix=peer_means,
        )
        return np.asarray(weights, dtype=float).reshape(n_peers)

    @staticmethod
    def audit(
        target: ModelUnit,
        peers: list[ModelUnit],
        X_fit,
        theta_fit,
        X_eval,
        theta_eval,
        intervention: Intervention,
        s_fit: int = 4,
        s_eval: int = 8,
        variance_penalty: float = 1.0,
        seed_offset: int = 0,
    ) -> StableAuditResult:
        #get repated resonses
        y_fit, Y_fit = StableDISCOSolver._collect_repeated_responses(
            target=target,
            peers=peers,
            X=X_fit,
            thetas=theta_fit,
            intervention=intervention,
            repeats=s_fit,
            seed_offset=seed_offset,
        )
        #get the average 
        y_fit_mean = y_fit.mean(axis=1)
        Y_fit_mean = Y_fit.mean(axis=1)
        # get the synthetic 
        weights = StableDISCOSolver.solve_stable_weights(
            target_means=y_fit_mean,
            peer_means=Y_fit_mean,
        )
        # collect responses again now with different seed. 
        y_eval, Y_eval = StableDISCOSolver._collect_repeated_responses(
            target=target,
            peers=peers,
            X=X_eval,
            thetas=theta_eval,
            intervention=intervention,
            repeats=s_eval,
            seed_offset=seed_offset + 100_000,
        )

        residual_samples = y_eval - np.einsum("nsp,p->ns", Y_eval, weights)
        mean_residuals = residual_samples.mean(axis=1)
        #Ut
        stable_magnitude = float(np.mean(np.abs(mean_residuals)))
        #Vt
        stability_variance = float(np.mean(np.var(residual_samples, axis=1, ddof=0)))
        stable_score = float(
            stable_magnitude - variance_penalty * np.sqrt(max(stability_variance, 0.0))
        )
        #get naive ut 
        naive_y = y_fit[:, 0]
        naive_Y = Y_fit[:, 0, :]
        naive_score, _ = DISCOSolver.solve_weights_and_distance(naive_y, naive_Y)

        return StableAuditResult(
            weights=weights,
            naive_score=float(naive_score),
            stable_magnitude=stable_magnitude,
            stability_variance=stability_variance,
            stable_score=stable_score,
            mean_residuals=mean_residuals,
            residual_samples=residual_samples,
        )