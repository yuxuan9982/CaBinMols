# graphgps/loader/custom_dataset.py
import numpy as np

import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from graphgps.transform.posenc_stats import compute_posenc_stats
from tqdm import tqdm
from rdkit import Chem

class SmilesDataset(InMemoryDataset):
    def __init__(self, root='datasets', csv_path='NHC-cracker-zzy-v1.csv', smiles2graph=smiles2graph, transform=None, pre_transform=None, use_dE_triplet_as_feature=False):
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'custom')
        self.csv_path = osp.join(self.folder, csv_path)
        self.use_dE_triplet_as_feature = use_dE_triplet_as_feature

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        
    @property
    def raw_file_names(self):
        # return [self.csv_path]  # 你的CSV文件名
        return self.csv_path  # 你的CSV文件名
    
    @property
    def processed_file_names(self):
        # return ['data.pt']  # 处理后的文件名
        return 'data.pt'  # 处理后的文件名
    
    def process(self):
        # 读取CSV并转换为图数据
        data_df = pd.read_csv(self.csv_path)
        smiles_list = data_df['SMILES']
        
        # 定义要预测的性质
        # 如果使用 dE_triplet 作为输入特征，则从预测目标中移除它
        # if self.use_dE_triplet_as_feature:
        #     target_properties = ['vbur_ratio_vbur_vtot', 'dE_AuCl']
        # else:
        #     target_properties = ['vbur_ratio_vbur_vtot', 'dE_AuCl', 'dE_triplet']
        target_properties = ['vbur_ratio_vbur_vtot', 'dE_AuCl']

        print('Converting SMILES strings into graphs...')
        if self.use_dE_triplet_as_feature:
            print('Using dE_triplet as graph-level input feature (data.u)')
            print(f'Target properties: {target_properties}')
        else:
            print(f'Target properties: {target_properties}')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            # graph = Chem.MolFromSmiles(smiles)
            graph = self.smiles2graph(smiles)

            # 检查 dE_triplet 是否有效（如果用作特征）
            if self.use_dE_triplet_as_feature and pd.isna(data_df['dE_triplet'].iloc[i]):
                print(f"[Warning] Skipped sample {i} ({smiles}) — dE_triplet is NaN (required as input feature)")
                continue
            
            # 检查所有要预测的性质是否都有效（跳过有 NaN 的样本）
            missing_props = [prop for prop in target_properties if pd.isna(data_df[prop].iloc[i])]
            if missing_props:
                print(f"[Warning] Skipped sample {i} ({smiles}) — missing properties: {missing_props}")
                continue

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            
            # 将 dE_triplet 作为图级别特征添加到 data.u（如果启用）
            if self.use_dE_triplet_as_feature:
                data.u = torch.Tensor([data_df['dE_triplet'].iloc[i]]).to(torch.double)
            
            # 将所有要预测的性质组合成一个向量
            target_values = [data_df[prop].iloc[i] for prop in target_properties]
            data.y = torch.Tensor(target_values).to(torch.double)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        # 统计并打印 skip 了多少个有缺失的数据
        total_samples = len(smiles_list)
        processed_samples = len(data_list)
        skipped_samples = total_samples - processed_samples
        print(f"Skipped {skipped_samples} samples due to missing dE_triplet or target properties.")
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
    
    def get_idx_split(self):
        """Get dataset splits with 8:1:1 ratio if no split file exists.
        
        Returns:
            Dict with 'train', 'val', 'test' splits indices.
        """
        # 如果没有 split_file，就随机划分
        num_graphs = len(self)  # 假设你的数据集有 __len__ 方法
        
        # 生成随机索引排列
        indices = torch.randperm(num_graphs)
        
        # 计算划分点
        train_end = int(0.8 * num_graphs)
        val_end = train_end + int(0.1 * num_graphs)
        
        # 划分索引
        split_dict = {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:],
        }
        
        return split_dict
    
    

if __name__ == '__main__':
    dataset = CustomCSVDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())