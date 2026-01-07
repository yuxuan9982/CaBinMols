import argparse
import gzip
import os
import pdb
import pickle
import threading
import time
import warnings
from copy import deepcopy

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

import model_block
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended

from torch_geometric.graphgym.config import cfg
from GraphGPS.recover_model import load_trained_model,process
from torch_geometric.data import Data, Batch
from metrics import Evaluator

import gzip
import pickle

# 假设你知道迭代次数
iteration_num = 0  # 替换为实际的迭代次数
file_path = f'results/_0/0_sampled_mols.pkl.gz'

with gzip.open(file_path, 'rb') as file:
    sampled_molecules = pickle.load(file)

print(f"从 {file_path} 读取了 {len(sampled_molecules)} 个分子")