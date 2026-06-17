from copy import deepcopy

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from isqed.core import Intervention, ModelUnit

def sample_nonlinear_regression_data(

    n_samples: int,
    dim: int,
    noise_std: float = 0.1,
    random_state: int = 0,
):
    """Makes nonlinear regression data"""
    
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, dim))

    y = (
        1.25 * np.sin(X[:, 0])
        + 0.75 * X[:, 1] ** 2
        - 0.90 * X[:, 2] * X[:, 3]
        + 0.50 * np.tanh(X[:, 4])
    )
    if dim > 5:
        y += 0.15 * np.sum(X[:, 5:], axis=1)

    y += rng.normal(scale=noise_std, size=n_samples)
    return X, y

class GaussianFeatureNoiseIntervention(Intervention):
    """inputs seeded noise into a x,
    this is to do the active auditing"""

    def apply(self, x, theta, seed=None):
        x_arr = np.asarray(x, dtype=float)
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(int(seed))
        return x_arr + rng.normal(scale=theta, size=x_arr.shape)


class BoostedTreeRegressorModel(ModelUnit):
    """the core model, able to add noise to outputs for sybil behavior"""
#gets regressor, i use boosted tree model. but could be other
    def __init__(
        self,
        regressor,
        name: str,
        output_noise_std: float = 0.0,
        bias_shift: float = 0.0,
        noise_distribution: str = "gaussian",
        noise_df: int = 3,
    ):
        super().__init__(name=name)
        self.regressor = regressor
        self.output_noise_std = output_noise_std
        self.bias_shift = bias_shift
        self.noise_distribution = noise_distribution
        self.noise_df = noise_df
#you can make different types of noise, gaussian, bounded uniform, student t with df for tail heaviness
    def _sample_noise(self, rng, size):
        if self.output_noise_std <= 0:
            return np.zeros(size, dtype=float)

        if self.noise_distribution == "gaussian":
            return rng.normal(scale=self.output_noise_std, size=size)

        if self.noise_distribution == "bounded_uniform":
            bound = np.sqrt(3.0) * self.output_noise_std
            return rng.uniform(low=-bound, high=bound, size=size)

        if self.noise_distribution == "student_t":
            if self.noise_df <= 2:
                raise ValueError("noise_df must be > 2 for finite variance in student_t noise.")
            scale = self.output_noise_std / np.sqrt(self.noise_df / (self.noise_df - 2.0))
            return rng.standard_t(df=self.noise_df, size=size) * scale

        raise ValueError(
            "Unsupported noise_distribution. Use one of: "
            "'gaussian', 'bounded_uniform', 'student_t'."
        )
# you can predict without noise 
    def _predict_core(self, input_data):
        X = np.asarray(input_data, dtype=float)
        X_2d = np.atleast_2d(X)
        preds = self.regressor.predict(X_2d) + self.bias_shift
        if X.ndim == 1:
            return float(preds[0])
        return preds

    def _forward(self, input_data):
        return self.predict_with_seed(input_data, seed=None)
#you can predict with noise 
    def predict_with_seed(self, input_data, seed=None):
        preds = self._predict_core(input_data)
        preds_arr = np.asarray(preds, dtype=float)

        if self.output_noise_std <= 0:
            return preds_arr

        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(int(seed) + 17_171)

        noise = self._sample_noise(rng, preds_arr.shape)
        noisy_preds = preds_arr + noise
        if np.isscalar(preds) or noisy_preds.ndim == 0:
            return float(np.asarray(noisy_preds).reshape(-1)[0])
        return noisy_preds
    #you can predict with noise where the seeds changes 
    def predict_batch_with_seeds(self, input_data, seeds=None):
        preds_arr = np.asarray(self._predict_core(input_data), dtype=float)
        if self.output_noise_std <= 0:
            return preds_arr

        if seeds is None:
            rng = np.random.default_rng()
            return preds_arr + self._sample_noise(rng, preds_arr.shape)

        noise = np.array(
            [
                self._sample_noise(np.random.default_rng(int(seed) + 17_171), size=1).reshape(-1)[0]
                for seed in seeds
            ],
            dtype=float,
        )
        return preds_arr + noise

def _fit_hist_gradient_booster(X_train, y_train, random_state: int):
    """ fits a boosted tree regressor"""
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.06,
        max_iter=120,
        l2_regularization=1e-3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def build_boosted_tree_ecosystem(
    
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_models: int,
    bootstrap_fraction: float = 0.75,
    base_seed: int = 0,
):
    """ bouilds a ecosysteom of boosted tree models, each trained on a different bootstrap sample of the data"""
    rng = np.random.default_rng(base_seed)
    models = []
    n_samples = len(X_train)
    subset_size = max(64, int(bootstrap_fraction * n_samples))

    for idx in range(n_models):
        subset_idx = rng.choice(n_samples, size=subset_size, replace=True)
        model = _fit_hist_gradient_booster(
            X_train[subset_idx],
            y_train[subset_idx],
            random_state=base_seed + idx,
        )
        models.append(
            BoostedTreeRegressorModel(
                regressor=model,
                name=f"TreeModel_{idx}",
            )
        )

    return models


def make_randomness_sybil(
    base_model: BoostedTreeRegressorModel,
    output_noise_std: float,
    name: str,
    noise_distribution: str = "gaussian",
    noise_df: int = 3,
):
    """make a sybil that adds noise to the outputs of a base model, so it is should cheat the ut score but not the new score"""
    cloned_regressor = deepcopy(base_model.regressor)
    return BoostedTreeRegressorModel(
        regressor=cloned_regressor,
        name=name,
        output_noise_std=output_noise_std,
        bias_shift=base_model.bias_shift,
        noise_distribution=noise_distribution,
        noise_df=noise_df,
    )