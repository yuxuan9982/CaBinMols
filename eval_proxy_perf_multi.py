import pandas as pd
import torch
import sys
import os
import numpy as np

# Assume gflownet.py is in the parent directory, and that all needed imports are accessible.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gflownet import Proxy

def main():
    # Arguments for proxy
    cfg_path = 'GraphGPS/configs/GPS/a-mols.yaml'
    ckpt_path = 'GraphGPS/results/models/model_best.pth'
    # ckpt_path = 'GraphGPS/results/models/model_best_v2_merged.pth'
    # ckpt_path = 'model_best.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    # data_path = './GraphGPS/datasets/custom/NHC-cracker-zzy-v1.csv'
    data_path = './GraphGPS/datasets/custom/NHC-cracker-zzy-v1.csv'
    df = pd.read_csv(data_path)

    # Targets to evaluate (multi-objective supported)
    target_cols = ['dE_triplet', 'vbur_ratio_vbur_vtot', 'dE_AuCl']
    # Example for multi-objective:
    # target_cols = ['dE_triplet', 'vbur_ratio_vbur_vtot', 'target_c']

    # Ensure expected columns exist
    assert 'SMILES' in df.columns, "CSV must contain a 'SMILES' column"
    for col in target_cols:
        assert col in df.columns, f"CSV must contain a '{col}' column"

    # Drop any rows where any target is NaN or missing, and reset index
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    smiles_list = df['SMILES'].tolist()
    true_targets = df[target_cols].values.astype(float)

    # Batch size for proxy (adjust as needed)
    batch_size = 64

    # Load proxy model
    proxy = Proxy(cfg_path, ckpt_path, device, reward_target_idx=None, num_tasks=len(target_cols))

    # Get proxy prediction in batches
    proxy_preds = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            preds = proxy(batch_smiles, return_vector=True)
            # Convert torch tensor or numpy array to flat list
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().numpy()
            elif hasattr(preds, 'cpu'):  # e.g., numpy with .cpu is not expected but just in case
                preds = preds.cpu().numpy()
            if isinstance(preds, (list, tuple)):
                preds = np.array(preds)
            proxy_preds.append(np.array(preds))

    proxy_preds = np.concatenate(proxy_preds, axis=0)

    # Align prediction shape to (N, num_targets)
    if proxy_preds.ndim == 1:
        proxy_preds = proxy_preds.reshape(-1, 1)
    if true_targets.ndim == 1:
        true_targets = true_targets.reshape(-1, 1)

    num_samples = min(len(proxy_preds), len(true_targets))
    proxy_preds = proxy_preds[:num_samples]
    true_targets = true_targets[:num_samples]

    # Compute difference and statistics per target
    diff = proxy_preds - true_targets
    abs_diff = np.abs(diff)
    mae = abs_diff.mean(axis=0)
    rmse = np.sqrt((diff ** 2).mean(axis=0))

    print(f"Number of samples: {num_samples}")
    for idx, col in enumerate(target_cols):
        print(f"[{col}] MAE: {mae[idx]:.4f} | RMSE: {rmse[idx]:.4f}")

    # Optionally: print a few rows for visual inspection
    # INSERT_YOUR_CODE
    thresholds = {
        'dE_triplet': 3.4,
    }
    for idx, col in enumerate(target_cols):
        if col in thresholds:
            threshold = thresholds[col]
            proxy_ge = int((proxy_preds[:, idx] >= threshold).sum())
            truth_ge = int((true_targets[:, idx] >= threshold).sum())
            both_ge = int(((proxy_preds[:, idx] >= threshold) & (true_targets[:, idx] >= threshold)).sum())
            print(f"[{col}] Number of proxy predictions >= {threshold}: {proxy_ge}")
            print(f"[{col}] Number of ground truths >= {threshold}: {truth_ge}")
            print(f"[{col}] Number of both proxy and truth >= {threshold}: {both_ge}")

    for i in range(200, min(230, num_samples)):
        truth_vals = ", ".join([f"{true_targets[i, j]:.4f}" for j in range(len(target_cols))])
        pred_vals = ", ".join([f"{proxy_preds[i, j]:.4f}" for j in range(len(target_cols))])
        diff_vals = ", ".join([f"{diff[i, j]:.4f}" for j in range(len(target_cols))])
        print(f"SMILES: {smiles_list[i]}, Truth: [{truth_vals}], Proxy: [{pred_vals}], Diff: [{diff_vals}]")

if __name__ == "__main__":
    main()
