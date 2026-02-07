"""
GraphGPS 模型推理脚本，支持 a-mols / a-mols_with_dE 配置。
与 smiles_dataset 的预处理保持一致，含 LapPE、可选 data.u、目标反归一化。
"""
import os
import sys

# 关键点：
# GraphGPS/graphgps 包内部大量使用绝对导入 `from graphgps...`
# 因此这里必须确保 `GraphGPS/` 目录在 sys.path 里，然后用 `import graphgps`
# 而不是 `import GraphGPS.graphgps`（后者会导致找不到顶层包名 `graphgps`）。
_GRAPHGPS_ROOT = os.path.dirname(__file__)  # .../CaBinMols/GraphGPS
if _GRAPHGPS_ROOT not in sys.path:
    sys.path.insert(0, _GRAPHGPS_ROOT)

import torch
import logging
import argparse
import pickle
import numpy as np
import graphgps  # noqa
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.data import Data
from ogb.utils import smiles2graph
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from graphgps.transform.posenc_stats import get_lap_decomp_stats
import io


def load_trained_model(cfg_path, checkpoint_path, device=None, num_tasks=None):
    """加载 GraphGPS 模型结构与权重。"""
    set_cfg(cfg)
    args = argparse.Namespace(cfg_file=cfg_path, opts=[])
    load_cfg(cfg, args)

    if cfg.accelerator == 'auto':
        cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_model()
    model.to(cfg.accelerator)

    ckpt = torch.load(checkpoint_path, map_location=cfg.accelerator)
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    assert num_tasks is not None
    # 设置模型输出维度
    cfg.share.dim_out = num_tasks
    model.load_state_dict(state_dict)
    model.eval()

    logging.info(f"Loaded model from {checkpoint_path} to device {cfg.accelerator}")
    return model


def load_norm_stats(root='datasets', csv_path='NHC-cracker-zzy-v1.csv'):
    """加载 target 归一化统计量，用于反归一化预测结果。"""
    # processed_dir = datasets/custom/processed
    folder = os.path.join(root, 'custom')
    processed_dir = os.path.join(folder, 'processed')
    norm_stats_path = os.path.join(processed_dir, 'target_norm_stats.pkl')
    if not os.path.exists(norm_stats_path):
        return None, None
    with open(norm_stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats['mean'], stats['std']


def denormalize(y_pred, target_mean, target_std):
    """将归一化后的预测值反归一化到原始尺度。"""
    if target_mean is None or target_std is None:
        return y_pred
    y = np.asarray(y_pred)
    out = y * target_std + target_mean
    if torch.is_tensor(y_pred):
        return torch.from_numpy(out.astype(np.float64)).to(device=y_pred.device)
    return out


def add_lap_pe(data, cfg):
    """根据 cfg 生成与 compute_posenc_stats 一致的 LapPE 特征。"""
    pecfg = getattr(cfg, 'posenc_LapPE', None)
    eigen = getattr(pecfg, 'eigen', None) if pecfg else None
    ln = getattr(eigen, 'laplacian_norm', 'none') if eigen else 'none'
    laplacian_norm = None if str(ln).lower() == 'none' else str(ln).lower()
    eigvec_norm = getattr(eigen, 'eigvec_norm', 'L2') if eigen else 'L2'
    max_freqs = getattr(eigen, 'max_freqs', 1) if eigen else 1

    num_nodes = data.num_nodes
    L = to_scipy_sparse_matrix(
        *get_laplacian(data.edge_index, normalization=laplacian_norm, num_nodes=num_nodes)
    ).toarray()
    evals, evects = np.linalg.eigh(L)
    EigVals, EigVecs = get_lap_decomp_stats(
        evals=evals, evects=evects, max_freqs=max_freqs, eigvec_norm=eigvec_norm
    )
    data.EigVals = EigVals
    data.EigVecs = EigVecs
    return data


def process(smiles, dE_triplet=None, device=None):
    """
    将 SMILES 转为与 smiles_dataset 一致的图数据。
    Args:
        smiles: SMILES 字符串
        dE_triplet: 可选，当 use_dE_triplet_as_feature=True 时必填
        device: 目标设备
    """
    use_dE = getattr(cfg.dataset, 'use_dE_triplet_as_feature', False)
    if use_dE and dE_triplet is None:
        raise ValueError("a-mols_with_dE 模型需要提供 --dE_triplet")

    data = Data()
    graph = smiles2graph(smiles)
    assert len(graph['edge_feat']) == graph['edge_index'].shape[1]
    assert len(graph['node_feat']) == graph['num_nodes']

    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)

    if use_dE:
        data.u = torch.tensor([[float(dE_triplet)]], dtype=torch.float64)

    data = add_lap_pe(data, cfg)
    if device is not None:
        data = data.to(device)
    return data


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        os.environ["PYTHONUTF8"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--smiles', type=str, default='COc1cccc2c1N(C)CO2')
    parser.add_argument('--dE_triplet', type=float, default=None,
                        help='a-mols_with_dE 模型推理时必填')
    parser.add_argument('--dataset_root', type=str, default='datasets')
    args = parser.parse_args()

    cfg_file = args.cfg
    ckpt = args.ckpt or os.path.join(
        os.path.dirname(cfg_file).replace("configs", "results"),
        os.path.splitext(os.path.basename(cfg_file))[0],
        "0/model_best.pth"
    )

    model = load_trained_model(cfg_file, ckpt, device=args.device)
    print("✅ Model loaded and ready for inference.")

    csv_path = getattr(cfg.dataset, 'csv_path', 'NHC-cracker-zzy-v1.csv')
    target_mean, target_std = load_norm_stats(root=args.dataset_root, csv_path=csv_path)
    if target_mean is not None:
        print(f"✅ Loaded target norm stats for denormalization")

    batch = process(args.smiles, dE_triplet=args.dE_triplet, device=cfg.accelerator)
    print("Input:", batch)

    with torch.no_grad():
        output = model(batch)

    output_orig = denormalize(output, target_mean, target_std)
    print("Output (normalized):", output)
    print("Output (denormalized):", output_orig)
    print("Target names: dE_triplet, vbur_ratio_vbur_vtot, dE_AuCl")
