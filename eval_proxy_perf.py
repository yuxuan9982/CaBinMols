import pandas as pd
import torch
import sys
import os

# Assume gflownet.py is in the parent directory, and that all needed imports are accessible.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gflownet import Proxy

def main():
    # Arguments for proxy
    cfg_path = 'GraphGPS/configs/GPS/a-mols.yaml'
    # ckpt_path = 'GraphGPS/results/models/model_best.pth'
    ckpt_path = 'GraphGPS/results/models/model_best_v2_merged.pth'
    # ckpt_path = 'model_best.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    # data_path = './GraphGPS/datasets/custom/NHC-cracker-zzy-v1.csv'
    data_path = './GraphGPS/datasets/custom/NHC-cracker-zzy-v2.csv'
    df = pd.read_csv(data_path)

    # Ensure expected columns exist
    assert 'SMILES' in df.columns, "CSV must contain a 'smiles' column"
    assert 'dE_triplet' in df.columns, "CSV must contain a 'dE_triplet' column"
    # Drop any rows where dE_triplet is NaN or missing, and reset index
    df = df[df['dE_triplet'].notna()].reset_index(drop=True)

    smiles_list = df['SMILES'].tolist()
    true_de_triplet = df['dE_triplet'].values

    # Batch size for proxy (adjust as needed)
    batch_size = 64

    # Load proxy model
    proxy = Proxy(cfg_path, ckpt_path, device)

    # Get proxy prediction in batches
    proxy_preds = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            preds = proxy(batch_smiles)
            # Convert torch tensor or numpy array to flat list
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().numpy()
            elif hasattr(preds, 'cpu'):  # e.g., numpy with .cpu is not expected but just in case
                preds = preds.cpu().numpy()
            # Now flatten if necessary (np.ndarray or torch), else if list just flatten
            import numpy as np
            if isinstance(preds, (list, tuple)):
                preds = np.array(preds)
            preds = np.ravel(preds)  # flatten
            proxy_preds.extend(preds.tolist())

    proxy_preds = proxy_preds[:len(true_de_triplet)]  # just in case

    # Compute difference and some statistics
    diff = [float(a) - float(b) for a, b in zip(proxy_preds, true_de_triplet)]
    abs_diff = [abs(d) for d in diff]
    mae = sum(abs_diff) / len(abs_diff)
    rmse = (sum((a-b)**2 for a, b in zip(proxy_preds, true_de_triplet)) / len(true_de_triplet)) ** 0.5

    print(f"Number of samples: {len(true_de_triplet)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Optionally: print a few rows for visual inspection
    # INSERT_YOUR_CODE
    threshold = 3.4
    proxy_ge_34 = sum([p >= threshold for p in proxy_preds])
    truth_ge_34 = sum([t >= threshold for t in true_de_triplet])
    both_ge_34 = sum([(p >= threshold) and (t >= threshold) for p, t in zip(proxy_preds, true_de_triplet)])

    print(f"Number of proxy predictions >= {threshold}: {proxy_ge_34}")
    print(f"Number of ground truths >= {threshold}: {truth_ge_34}")
    print(f"Number of both proxy and truth >= {threshold}: {both_ge_34}")
    for i in range(200,min(230, len(true_de_triplet))):
        print(f"SMILES: {smiles_list[i]}, Truth: {true_de_triplet[i]:.4f}, Proxy: {proxy_preds[i]:.4f}, Diff: {diff[i]:.4f}")

if __name__ == "__main__":
    main()
