import torch
import logging
import os
import argparse
import numpy as np
import graphgps  # noqa
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.data import Data, InMemoryDataset, download_url
from ogb.utils import smiles2graph
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

import sys


def load_trained_model(cfg_path, checkpoint_path, device=None):
    """
    仅加载 GraphGPS 模型结构与权重。
    """
    # 1. 初始化配置
    set_cfg(cfg)
    args = argparse.Namespace(cfg_file=cfg_path, opts=[])
    load_cfg(cfg, args)

    # 2. 自动修正 accelerator 字段
    if cfg.accelerator == 'auto':
        cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 如果用户手动指定 device，就覆盖
    if device is not None:
        cfg.accelerator = device

    # 3. 创建模型
    model = create_model()
    model.to(cfg.accelerator)

    # 4. 加载 state_dict
    ckpt = torch.load(checkpoint_path, map_location=cfg.accelerator)
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    logging.info(f"Loaded model from {checkpoint_path} to device {cfg.accelerator}")
    return model


def process(smiles, device=None):
    # 读取CSV并转换为图数据
    data = Data()

    # graph = Chem.MolFromSmiles(smiles)
    graph = smiles2graph(smiles)


    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])

    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(
        torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
        torch.int64)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.long)
    data = add_lap_pe(data)
    if device is not None:
        data = data.to(device)
    return data


from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_dense_adj
import torch



def add_lap_pe(data, k=1, normalization="sym", eigvec_norm="L2"):
    """
    为 GraphGPS 模型生成与 compute_posenc_stats() 一致的 LapPE 特征。
    会添加:
        data.EigVals: [num_nodes, k, 1]
        data.EigVecs: [num_nodes, k]
        data.pe_LapPE: [num_nodes, k]
    """
    num_nodes = data.num_nodes
    L = to_scipy_sparse_matrix(
        *get_laplacian(data.edge_index, normalization=normalization, num_nodes=num_nodes)
    ).toarray()
    evals, evects = np.linalg.eigh(L)
    idx = evals.argsort()[:k]
    evals, evects = evals[idx], np.real(evects[:, idx])

    evals = torch.from_numpy(np.real(evals)).float().clamp_min(0)
    evects = torch.from_numpy(evects).float()

    # --- GraphGPS 的 eigvec_normalizer 实现 ---
    denom = evects.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
    evects = evects / denom

    # pad 逻辑（避免小图）
    if num_nodes < k:
        EigVecs = F.pad(evects, (0, k - num_nodes), value=float('nan'))
    else:
        EigVecs = evects
    if num_nodes < k:
        EigVals = F.pad(evals, (0, k - num_nodes), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(num_nodes, 1).unsqueeze(2)

    data.EigVals = EigVals
    data.EigVecs = EigVecs

    return data



import time
import io
if __name__ == "__main__":
    print(sys.platform)
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        os.environ["PYTHONUTF8"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--device', type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    cfg_file = args.cfg
    ckpt = args.ckpt or os.path.join(
        os.path.dirname(cfg_file).replace("configs", "results"),
        os.path.splitext(os.path.basename(cfg_file))[0],
        "0/model_best.pth"
    )

    model = load_trained_model(cfg_file, ckpt, device=args.device)
    print("✅ Model successfully loaded and ready for inference.")

    # batch = process(input("Input Smiles:"))
    batch = process("COc1cccc2c1N(C)CO2")
    # batch = add_lap_pe(batch)
    batch = batch.to(cfg.accelerator)
    print(batch)
    with torch.no_grad():
        output = model(batch)
    print(output)

    # n_rounds = 100
    # start = time.time()
    # for i in range(n_rounds):
    #     batch = process("Cc1cc(C)c(N2CN(c3c(C)cc(C)cc3C)C(=O)C2=O)c(C)c1")
    #     batch = add_lap_pe(batch)
    #     batch = batch.to(cfg.accelerator)
    #     output = model(batch)
    # end = time.time()

    # elapsed = end - start
    # print(f"⏱️  {n_rounds} rounds finished in {elapsed:.4f} s")
    # print("Output sample:", output)
#python recover_model.py --cfg configs/GPS/a-mols.yaml --ckpt results/models/model_best.pth


