# experiments/exp16_stable_disco_detection_curves.py

import argparse
import os
import sys
import concurrent.futures 
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
    n_honest,
    sybil_noise_std,
    n_fit,
    n_eval,
    s_fit,
    s_eval_grid,
    s_eval_reference,
    variance_penalty,
    seed,
    sybil_noise_family,
    sybil_noise_df,
):
    """
    Experiment 2 (detection curves / sample complexity):
      - Fix ecosystem size and attack strength.
      - Sweep S_eval (repeated evaluation queries).
      - Keep a lock-then-evaluate pipeline:
            1) fit on denoised means with S_fit repeats,
            2) lock weights,
            3) evaluate residual variance using S_eval repeats.
    """
    X_train, y_train = sample_nonlinear_regression_data(
        n_samples=1_200,
        dim=dim,
        noise_std=0.10,
        random_state=seed,
    )

    ecosystem_models = build_boosted_tree_ecosystem(
        X_train=X_train,
        y_train=y_train,
        n_models=n_honest,
        bootstrap_fraction=0.75,
        base_seed=seed,
    )

    target = ecosystem_models[0]
    honest_peers = ecosystem_models[1:]

    sybil = make_randomness_sybil(
        base_model=honest_peers[0],
        output_noise_std=sybil_noise_std,
        name=f"Sybil_noise_{sybil_noise_std:.2f}",
        noise_distribution=sybil_noise_family,
        noise_df=sybil_noise_df,
    )

    intervention = GaussianFeatureNoiseIntervention()

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
        (target, honest_peers, "target", False),
        (honest_peers[0], honest_peers[1:], "normal_peer", False),
        (sybil, honest_peers[1:], "sybil", True),
    ]

    # Reference variance with high S_eval to approximate the large-sample limit.
    reference_stats = {}
    for model, comparison_peers, role, is_sybil in audit_targets:
        ref_audit = StableDISCOSolver.audit(
            target=model,
            peers=comparison_peers,
            X_fit=X_fit,
            theta_fit=theta_fit,
            X_eval=X_eval,
            theta_eval=theta_eval,
            intervention=intervention,
            s_fit=s_fit,
            s_eval=s_eval_reference,
            variance_penalty=variance_penalty,
            seed_offset=seed,
        )
        reference_stats[role] = {
            "reference_variance": ref_audit.stability_variance,
            "reference_stable_score": ref_audit.stable_score,
        }

    rows = []
    for s_eval in s_eval_grid:
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

            reference_var = reference_stats[role]["reference_variance"]

            rows.append(
                {
                    "trial": seed,
                    "s_eval": int(s_eval),
                    "s_fit": int(s_fit),
                    "model_name": model.name,
                    "role": role,
                    "is_sybil": int(is_sybil),
                    "sybil_noise_std": sybil_noise_std,
                    "sybil_noise_family": sybil_noise_family,
                    "sybil_noise_df": sybil_noise_df,
                    "naive_score": audit.naive_score,
                    "stable_magnitude": audit.stable_magnitude,
                    "stability_variance": audit.stability_variance,
                    "stable_score": audit.stable_score,
                    "reference_variance": reference_var,
                    "variance_abs_error": abs(audit.stability_variance - reference_var),
                    "reference_stable_score": reference_stats[role]["reference_stable_score"],
                }
            )

    return rows


def run_experiment(
    dim,
    n_honest,
    trials,
    n_fit,
    n_eval,
    s_fit,
    s_eval_reference,
    variance_penalty,
    sybil_noise_std,
    sybil_noise_family,
    sybil_noise_df,
    output_file,
):
    s_eval_grid = [2, 4, 8, 16, 32, 64]
    
    print(f"--- Running Exp 16: Detection Curves ---")
    print(f"Dim: {dim}, Honest models: {n_honest}, Trials: {trials}")
    print(f"Sybil noise family: {sybil_noise_family}")
    
    rows = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_single_trial,
                dim, n_honest, sybil_noise_std, n_fit, n_eval, s_fit,
                s_eval_grid, s_eval_reference, variance_penalty, trial_seed,
                sybil_noise_family, sybil_noise_df
            )
            for trial_seed in range(trials)
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=trials, desc="Simulating Detection Curves"):
            try:
                trial_rows = future.result()
                rows.extend(trial_rows)
            except Exception as e:
                print(f"A trial failed with exception: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stable DISCO detection-curve experiment")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n_honest", type=int, default=8)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--n_fit", type=int, default=40)
    parser.add_argument("--n_eval", type=int, default=60)
    parser.add_argument("--s_fit", type=int, default=16)
    parser.add_argument("--s_eval_reference", type=int, default=128)
    parser.add_argument("--variance_penalty", type=float, default=1.0)
    parser.add_argument("--sybil_noise_std", type=float, default=0.30)
    parser.add_argument(
        "--sybil_noise_family",
        type=str,
        default="gaussian",
        choices=["gaussian", "bounded_uniform", "student_t"],
    )
    parser.add_argument("--sybil_noise_df", type=int, default=3)
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/tables/exp16_stable_disco_detection_curves.csv",
    )
    args = parser.parse_args()

    run_experiment(
        dim=args.dim,
        n_honest=args.n_honest,
        trials=args.trials,
        n_fit=args.n_fit,
        n_eval=args.n_eval,
        s_fit=args.s_fit,
        s_eval_reference=args.s_eval_reference,
        variance_penalty=args.variance_penalty,
        sybil_noise_std=args.sybil_noise_std,
        sybil_noise_family=args.sybil_noise_family,
        sybil_noise_df=args.sybil_noise_df,
        output_file=args.output_file,
    )