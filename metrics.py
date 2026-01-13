import gzip
import pickle
import rdkit.DataStructs
from rdkit import Chem
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
import heapq
import csv

def get_tanimoto_pairwise(mols):
    fps = [Chem.RDKFingerprint(i.mol) for i in mols]
    pairwise_sim = []
    for i in range(len(mols)):
        pairwise_sim.extend(rdkit.DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    return pairwise_sim


class NumModes:
    def __init__(self, reward_exp, reward_norm, reward_thr=8, tanimoto_thr=0.7):
        self.reward_exp = reward_exp
        self.reward_norm = reward_norm
        self.reward_thr = reward_thr
        self.tanimoto_thr = tanimoto_thr
        self.modes = []
        self.max_reward = -1000
    def __call__(self, batch):
        candidates = []
        for some in batch:
            reward, mol = some[0], some[1]
            reward = (reward ** (1/self.reward_exp)) * self.reward_norm
            if reward > self.max_reward: 
                self.max_reward = reward
            if reward > self.reward_thr:
                candidates.append(mol)
        if len(candidates) > 0:
            # add one mode if needed
            if len(self.modes)==0: 
                self.modes.append(Chem.RDKFingerprint(candidates[0].mol))
            for mol in candidates:
                fp = Chem.RDKFingerprint(mol.mol)
                sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))
                if all(sims < self.tanimoto_thr):
                    self.modes.append(fp)
        return self.max_reward, len(self.modes)
    def add(self, mols):
        candidates = []
        reward, mol = mols[0], mols[1]
        if reward > self.max_reward: 
            self.max_reward = reward
        if reward > self.reward_thr:
            candidates.append(mol)
        if len(candidates) > 0:
            if len(self.modes)==0: 
                self.modes.append(Chem.RDKFingerprint(candidates[0].mol))
            for mol in candidates:
                fp = Chem.RDKFingerprint(mol.mol)
                sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))
                if all(sims < self.tanimoto_thr):
                    self.modes.append(fp)
    def get_modes(self):
        return len(self.modes)


# class Top1000:
#     def __init__(self):
#         self.min_heap = []  # 最小堆，存储前100大的元素
#         self.size = 1000

#     def add(self, num):
#         if len(self.min_heap) < self.size:
#             heapq.heappush(self.min_heap, num)
#         else:
#             if num > self.min_heap[0]:
#                 heapq.heapreplace(self.min_heap, num)

#     def get_top_1000(self):
#         return sorted(self.min_heap, reverse=True)
class _HeapItem:
    """Wrapper class to ensure only the reward (r) is compared."""
    def __init__(self, r, m):
        self.r = r
        self.m = m

    def __lt__(self, other):
        return self.r < other.r  # Only compare `r`, ignore `m`

class Top1000:
    def __init__(self):
        self.min_heap = []  # Stores _HeapItem objects
        self.size = 1000

    def add(self, item):
        r, m = item
        heap_item = _HeapItem(r, m)  # Wrap in our custom class
        if len(self.min_heap) < self.size:
            heapq.heappush(self.min_heap, heap_item)
        else:
            if r > self.min_heap[0].r:  # Compare against the smallest `r` in heap
                heapq.heapreplace(self.min_heap, heap_item)

    def get_top_1000(self):
        # Return sorted list of (r, m) tuples, descending by `r`
        sorted_items = sorted(self.min_heap, key=lambda x: -x.r)
        return [(item.r, item.m) for item in sorted_items]


# def eval_mols(mols, reward_norm=8, reward_exp=10, algo="gfn"):
#     def r2r_back(r):
#         return r ** (1. / reward_exp) * reward_norm
    
#     numModes_above_7_5 = NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=7.5)
#     _, num_modes_above_7_5 = numModes_above_7_5(mols)
#     numModes_above_8_0 = NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=8.)
#     _, num_modes_above_8_0 = numModes_above_8_0(mols)

    

#     top_ks = [10, 100, 1000]
#     avg_topk_rs = {}
#     avg_topk_tanimoto = {}
#     mol_r_map = {}

#     for i in range(len(mols)):
#         if algo == 'gfn':
#             r, m, trajectory_stats, inflow = mols[i]
#         else:
#             r, m = mols[i]
#         r = r2r_back(r)
#         mol = Chem.MolFromSmiles(m.smiles)  # 使用你感兴趣的分子的SMILES表示
#         scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # 获取Bemis-Murcko骨架
#         scaffold_smiles = Chem.MolToSmiles(scaffold)
#         mol_r_map[scaffold_smiles] = r
    
#     unique_rs = list(mol_r_map.values())
#     unique_rs = sorted(unique_rs, reverse=True)
#     unique_rs = np.array(unique_rs)
#     num_above_7_5 = np.sum(unique_rs > 7.5) # just a integer
#     num_above_8_0 = np.sum(unique_rs > 8.0)

#     sorted_mol_r_map = sorted(mol_r_map.items(), key=lambda kv: kv[1], reverse=True)
#     for top_k_idx, top_k in enumerate(top_ks):
#         avg_topk_rs[top_k] = np.mean(unique_rs[:top_k])
        
#         topk_mols = [mol for (mol, r) in sorted_mol_r_map[:top_k]]
#         avg_topk_tanimoto[top_k] = np.mean(get_tanimoto_pairwise(topk_mols))

#     return avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, num_above_7_5, num_above_8_0

def verify_csv_pkl_files(csv_path="molecule_results_n.csv", pkl_path="molecule_objects.pkl.gz", verbose=True):
    """
    独立的验证函数，验证CSV文件和pickle文件是否对应
    
    Args:
        csv_path: CSV文件路径
        pkl_path: pickle文件路径
        verbose: 是否打印详细信息
        
    Returns:
        bool: 如果所有记录都匹配返回True，否则返回False
    """
    try:
        # 读取CSV文件
        csv_smiles = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 1:
                    csv_smiles.append(row[0])  # smiles在第一列
        
        # 读取pickle文件
        pkl_data = []
        with gzip.open(pkl_path, "rb") as f:
            while True:
                try:
                    index, mol_obj = pickle.load(f)
                    pkl_data.append((index, mol_obj.smiles))
                except EOFError:
                    break
        
        # 验证数量是否一致
        if len(csv_smiles) != len(pkl_data):
            if verbose:
                print(f"❌ 数量不匹配: CSV有{len(csv_smiles)}行, pickle有{len(pkl_data)}个对象")
            return False
        
        # 验证每个索引和smiles是否匹配
        mismatches = []
        for i, (index, pkl_smiles) in enumerate(pkl_data):
            if index != i:
                mismatches.append(f"索引不匹配: 期望{i}, 实际{index}")
            if i < len(csv_smiles) and csv_smiles[i] != pkl_smiles:
                mismatches.append(f"第{i}行: CSV smiles='{csv_smiles[i]}', pickle smiles='{pkl_smiles}'")
        
        if mismatches:
            if verbose:
                print(f"❌ 发现{len(mismatches)}个不匹配:")
                for mismatch in mismatches[:10]:  # 只显示前10个
                    print(f"  - {mismatch}")
                if len(mismatches) > 10:
                    print(f"  ... 还有{len(mismatches)-10}个不匹配")
            return False
        
        if verbose:
            print(f"✅ 验证通过: {len(csv_smiles)}条记录全部匹配")
        return True
        
    except FileNotFoundError as e:
        if verbose:
            print(f"❌ 文件未找到: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ 验证过程出错: {e}")
        return False


class Evaluator:
    def __init__(self,reward_norm=8, reward_exp=10, algo="gfn"):
        self.numModes_above_7_5=NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=3.5)
        self.numModes_above_8_0=NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=4.0)
        self.top_ks = [10, 100, 1000]
        self.avg_topk_tanimoto = {}
        self.mol_r_map = {}
        self.reward_norm=reward_norm
        self.reward_exp=reward_exp
        self.algo=algo
        self.map_cnt_7_5=0
        self.map_cnt_8_0=0
        self.top1000 = Top1000()
        with open("molecule_results_n.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "reward", "core"])
        # 初始化分子对象保存文件（使用gzip压缩以减小文件大小）
        self.mol_file = gzip.open("molecule_objects.pkl.gz", "wb")
        self.mol_file_index = 0  # 用于记录写入顺序，方便与CSV对应
        self.avg_core = [0 for _ in range(32)]
        self.cnt_core = [0 for _ in range(32)]

    def r2r_back(self,r):
        return r ** (1. / self.reward_exp) * self.reward_norm
    def add(self,batch):
        for i in range(len(batch)):
            if self.algo == 'gfn':
                r, m, action_states, inflow = batch[i]
            else:
                r, m = batch[i]
            r = self.r2r_back(r)
            mol = Chem.MolFromSmiles(m.smiles)  # 使用你感兴趣的分子的SMILES表示
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # 获取Bemis-Murcko骨架
            scaffold_smiles = Chem.MolToSmiles(scaffold)

            if scaffold_smiles not in self.mol_r_map or self.mol_r_map[scaffold_smiles]<r:
                self.mol_r_map[scaffold_smiles] = r
                if r>3.5:self.map_cnt_7_5+=1
                if r>4.0:self.map_cnt_8_0+=1
                self.top1000.add((r,m))
                
            self.numModes_above_7_5.add((r,m))
            self.numModes_above_8_0.add((r,m))

            assert action_states[0]-853 >=0 and action_states[0]-853 <32
            self.avg_core[action_states[0]-853] += r
            self.cnt_core[action_states[0]-853] += 1
            with open("molecule_results_n.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([m.smiles, r, action_states[0]-853])
            # 同时保存分子对象到pickle文件（使用gzip压缩）
            pickle.dump((self.mol_file_index, m), self.mol_file)
            self.mol_file.flush()  # 确保数据立即写入，避免丢失
            self.mol_file_index += 1
    
    def eval_mols(self):
        avg_topk_rs = {}
        avg_topk_tanimoto = {}
        top_lst = self.top1000.get_top_1000()
        for top_k_idx, top_k in enumerate(self.top_ks):
            num_values = [r for (r,mol) in top_lst[:top_k]]
            topk_mols = [mol for (r, mol) in top_lst[:top_k]]
            avg_topk_tanimoto[top_k] = np.mean(get_tanimoto_pairwise(topk_mols))
            avg_topk_rs[top_k] = np.mean(num_values[:top_k])
        num_modes_above_7_5= self.numModes_above_7_5.get_modes()
        num_modes_above_8_0= self.numModes_above_8_0.get_modes()
        num_above_7_5 = self.map_cnt_7_5
        num_above_8_0 = self.map_cnt_8_0
        print("Average reward per core:", [avg / cnt if cnt > 0 else 0 for avg, cnt in zip(self.avg_core, self.cnt_core)])
        print("Visit time per core:", [cnt for cnt in self.cnt_core])
        return avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, num_above_7_5, num_above_8_0
    
    def verify_csv_pkl_match(self, csv_path="molecule_results_n.csv", pkl_path="molecule_objects.pkl.gz", verbose=True):
        """
        验证CSV文件和pickle文件是否对应（调用独立验证函数）
        
        Args:
            csv_path: CSV文件路径
            pkl_path: pickle文件路径
            verbose: 是否打印详细信息
            
        Returns:
            bool: 如果所有记录都匹配返回True，否则返回False
        """
        return verify_csv_pkl_files(csv_path, pkl_path, verbose)
    
    def close(self):
        """关闭分子对象文件"""
        if hasattr(self, 'mol_file') and self.mol_file is not None:
            self.mol_file.close()
            self.mol_file = None
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()


# 使用示例：
# if __name__ == "__main__":
#     # 方式1: 使用独立函数验证
#     verify_csv_pkl_files("molecule_results_n.csv", "molecule_objects.pkl.gz")
#     
#     # 方式2: 使用Evaluator实例验证
#     evaluator = Evaluator()
#     evaluator.verify_csv_pkl_match()
#     evaluator.close()