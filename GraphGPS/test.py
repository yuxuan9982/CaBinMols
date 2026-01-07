import torch
import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.data import Data
from ogb.utils import smiles2graph

args = parse_args()
set_cfg(cfg)
load_cfg(cfg, args)
dump_cfg(cfg)
print(cfg.posenc_LapPE.enable)
auto_select_device()
# Set Pytorch environment
torch.set_num_threads(cfg.num_threads)
checkpoint = torch.load("99.ckpt")
model = create_model(dim_in=1,dim_out=8)
model.load_state_dict(checkpoint['model_state'])
model.eval()
smiles = 'CC(=O)Nc1c(-c2ccccc2)c(C)nn1-c1ccc(C(=O)Nc2cccnc2)cc1'
graph = smiles2graph(smiles)
data = Data()
data.__num_nodes__ = int(graph['num_nodes'])
data.edge_index = torch.from_numpy(graph['edge_index']).to(
    torch.int64)
data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
    torch.int64)
data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
# from torch_geometric.utils import get_laplacian
# # 计算拉普拉斯矩阵的特征值和特征向量
# edge_index, edge_weight = get_laplacian(
#     data.edge_index, 
#     normalization='sym',  # 对称归一化
#     num_nodes=data.num_nodes
# )
# L = torch.sparse_coo_tensor(edge_index, edge_weight, (data.num_nodes, data.num_nodes)).to_dense().to(model.device)
# eigvals, eigvecs = torch.linalg.eigh(L)  # 计算特征分解

# # 存储到 data 中
# data.EigVals = eigvals.unsqueeze(1)
# data.EigVecs = eigvecs
# data = data.to(model.device)

# print("eigvals shape:", data.EigVals.shape)  # 应为 (num_nodes,)
# print("eigvecs shape:", data.EigVecs.shape)  # 应为 (num_nodes, num_nodes)

from graphgps.transform.posenc_stats import compute_posenc_stats

# 设定需要计算的PE类型（例如 LapPE）
pe_enabled_list = ['LapPE']  # 根据你的需求修改

# 检查图的 undirected 属性
is_undirected = data.is_undirected()

# 直接计算 PE 统计量
compute_posenc_stats(
    data,  # 直接传入单图数据
    pe_types=pe_enabled_list,
    is_undirected=is_undirected,
    cfg=cfg  # 确保传入正确的配置
)

data = data.to(model.device)

# 现在 data 应该包含 eigvals 和 eigvecs
print(data.EigVals.shape, data.EigVecs.shape)  # 检查形状
print(model(data)) 