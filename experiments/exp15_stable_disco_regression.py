# experiments/exp15_stable_disco_regression.py

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sisqed.stable_geometry import StableDISCOSolver
from sisqed.synthetic import (
    GaussianFeatureNoiseIntervention,
    build_boosted_tree_ecosystem,
    make_randomness_sybil,
    sample_nonlinear_regression_data,
)


def generate_probe_design(n_pairs, dim, theta_min, theta_max, seed):
    rng = np.random.default_rng(seed)
    X_probe = rng.normal(size=(n_pairs, dim))
    theta_probe = rng.uniform(theta_min, theta_max, size=n_pairs)
    return X_probe, theta_probe


def run_single_trial(
    dim,
    n_peers,
    sybil_noise_std,
    n_fit,
    n_eval,
    s_fit,
    s_eval,
    variance_penalty,
    seed,
):
    #make data 
    X_train, y_train = sample_nonlinear_regression_data(
        n_samples=1_200,
        dim=dim,
        noise_std=0.10,
        random_state=seed,
    )
#make models 
    ecosystem_models = build_boosted_tree_ecosystem(
        X_train=X_train,
        y_train=y_train,
        n_models=n_peers + 2,
        bootstrap_fraction=0.75,
        base_seed=seed,
    )
    # target could be unique but dosnt have to.
    target = ecosystem_models[0]
    peers = ecosystem_models[1 : n_peers + 1]
    sybil = make_randomness_sybil(
        #sybil is just a peer with noise 
        base_model=peers[0],
        output_noise_std=sybil_noise_std,
        name=f"Sybil_noise_{sybil_noise_std:.2f}",
    )

    intervention = GaussianFeatureNoiseIntervention()
    # generate query design
    X_fit, theta_fit = generate_probe_design(
        n_pairs=n_fit,
        dim=dim,
        theta_min=0.05,
        theta_max=0.35,
        seed=seed + 11,
    )
    X_eval, theta_eval = generate_probe_design(
        n_pairs=n_eval,
        dim=dim,
        theta_min=0.05,
        theta_max=0.35,
        seed=seed + 29,
    )

    audit_targets = [
        (target, peers, "target", False),
        (peers[0], [p for p in peers[1:]], "normal_peer", False),
        (sybil, peers[1:], "sybil", True),
    ]

    rows = []
    #audit
    for model, comparison_peers, role, is_sybil in audit_targets:
        audit = StableDISCOSolver.audit(
            target=model,
            peers=comparison_peers,
            X_fit=X_fit,
            theta_fit=theta_fit,
            X_eval=X_eval,
            theta_eval=theta_eval,
            intervention=intervention,
            s_fit=s_fit,
            s_eval=s_eval,
            variance_penalty=variance_penalty,
            seed_offset=seed,
        )

        weight_entropy = float(
            -np.sum(np.clip(audit.weights, 1e-12, 1.0) * np.log(np.clip(audit.weights, 1e-12, 1.0)))
        )

        rows.append(
            {
                "trial": seed,
                "model_name": model.name,
                "role": role,
                "is_sybil": int(is_sybil),
                "sybil_noise_std": sybil_noise_std,
                "naive_score": audit.naive_score,
                "stable_magnitude": audit.stable_magnitude,
                "stability_variance": audit.stability_variance,
                "stable_score": audit.stable_score,
                "max_weight": float(np.max(audit.weights)),
                "weight_entropy": weight_entropy,
                "mean_abs_residual": float(np.mean(np.abs(audit.mean_residuals))),
            }
        )

    return rows


def run_experiment(
    dim=8,
    n_peers=5,
    trials=20,
    n_fit=40,
    n_eval=60,
    s_fit=4,
    s_eval=8,
    variance_penalty=1.0,
    #should be run with diffferent penelties
):
    print("--- Running Exp 15: Stable DISCO under Randomness Sybils ---")
    print(f"Dim: {dim}, Peers: {n_peers}, Trials: {trials}")
    print(f"Fit pairs: {n_fit}, Eval pairs: {n_eval}, S_fit: {s_fit}, S_eval: {s_eval}")

    rows = []
    noise_grid = [0.0, 0.10, 0.20, 0.35, 0.50]

    for trial_seed in tqdm(range(trials), desc="Simulating Stable DISCO"):
        for sybil_noise_std in noise_grid:
            rows.extend(
                run_single_trial(
                    dim=dim,
                    n_peers=n_peers,
                    sybil_noise_std=sybil_noise_std,
                    n_fit=n_fit,
                    n_eval=n_eval,
                    s_fit=s_fit,
                    s_eval=s_eval,
                    variance_penalty=variance_penalty,
                    seed=trial_seed,
                )
            )

    df = pd.DataFrame(rows)

    output_dir = "results/tables"
    output_file = os.path.join(output_dir, "exp15_stable_disco_regression.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
    print("\nMean scores by role and Sybil noise:")
    print(
        df.groupby(["role", "sybil_noise_std"])[["naive_score", "stable_score", "stability_variance"]]
        .mean()
        .round(4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stable DISCO regression prototype")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n_peers", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--n_fit", type=int, default=40)
    parser.add_argument("--n_eval", type=int, default=60)
    parser.add_argument("--s_fit", type=int, default=4)
    parser.add_argument("--s_eval", type=int, default=8)
    parser.add_argument("--variance_penalty", type=float, default=1.0)
    args = parser.parse_args()

    run_experiment(
        dim=args.dim,
        n_peers=args.n_peers,
        trials=args.trials,
        n_fit=args.n_fit,
        n_eval=args.n_eval,
        s_fit=args.s_fit,
        s_eval=args.s_eval,
        variance_penalty=args.variance_penalty,
    )