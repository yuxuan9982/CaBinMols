import gzip
import pickle
import json
import os
import argparse
import sys
import rdkit.DataStructs
from rdkit import Chem
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
import heapq
import csv
from datetime import datetime

def get_tanimoto_pairwise(mols):
    fps = [Chem.RDKFingerprint(i.mol) for i in mols]
    pairwise_sim = []
    for i in range(len(mols)):
        pairwise_sim.extend(rdkit.DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    return pairwise_sim

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

class ObjectiveModes:
    def __init__(self, obj1_thr, obj2_thr=0.1, obj3_thr=-4.0, tanimoto_thr=0.7):
        self.obj1_thr = float(obj1_thr)
        self.obj2_thr = float(obj2_thr)
        self.obj3_thr = float(obj3_thr)
        self.tanimoto_thr = float(tanimoto_thr)
        self.modes = []

    def _hit(self, objective_values):
        if objective_values is None:
            return False
        values = np.asarray(objective_values, dtype=np.float64).reshape(-1)
        if values.shape[0] < 3:
            return False
        return (values[0] > self.obj1_thr) and (values[1] > self.obj2_thr) and (values[2] > self.obj3_thr)

    def add(self, mol, objective_values):
        if not self._hit(objective_values):
            return
        fp = Chem.RDKFingerprint(mol.mol)
        if len(self.modes) == 0:
            self.modes.append(fp)
            return
        sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))
        if all(sims < self.tanimoto_thr):
            self.modes.append(fp)

    def get_modes(self):
        return len(self.modes)

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
        mismatches = []
        max_show = 10
        n_checked = 0

        with open(csv_path, "r", encoding="utf-8", newline="") as cf, gzip.open(pkl_path, "rb") as pf:
            reader = csv.reader(cf)
            try:
                next(reader)  # 跳过表头
            except StopIteration:
                if verbose:
                    print("❌ CSV为空或缺少表头")
                return False

            csv_exhausted = False
            while True:
                try:
                    index, mol_obj = pickle.load(pf)
                except EOFError:
                    break

                # 获取CSV对应行
                try:
                    row = next(reader)
                except StopIteration:
                    csv_exhausted = True
                    mismatches.append(f"数量不匹配: pickle还有数据(当前index={index})但CSV已结束")
                    break

                # smiles在第一列
                csv_smiles = row[0] if len(row) >= 1 else ""
                pkl_smiles = getattr(mol_obj, "smiles", None)

                if index != n_checked:
                    mismatches.append(f"索引不匹配: 期望{n_checked}, 实际{index}")

                # 如果CSV有 mol_file_idx（第4列），也顺便核对一下
                if len(row) >= 4:
                    try:
                        csv_idx = int(row[3])
                        if csv_idx != index:
                            mismatches.append(f"mol_file_idx不匹配: 第{n_checked}行 CSV mol_file_idx={csv_idx}, pickle index={index}")
                    except Exception:
                        # 非整数就不校验
                        pass

                if csv_smiles != pkl_smiles:
                    mismatches.append(f"第{n_checked}行: CSV smiles='{csv_smiles}', pickle smiles='{pkl_smiles}'")

                n_checked += 1

                # 只保留前 max_show 个 mismatch 文本，避免内存膨胀
                if len(mismatches) > max_show:
                    mismatches = mismatches[:max_show]

            # pickle读完后，检查CSV是否还有剩余有效行
            if not csv_exhausted:
                for extra_row in reader:
                    if len(extra_row) == 0:
                        continue
                    if len(extra_row) >= 1 and str(extra_row[0]).strip() == "":
                        continue
                    mismatches.append(f"数量不匹配: CSV还有额外行(从第{n_checked}行开始)但pickle已结束")
                    break

        if mismatches:
            if verbose:
                print(f"❌ 验证失败: 已检查{n_checked}条，发现不匹配(仅展示前{max_show}条):")
                for mismatch in mismatches[:max_show]:
                    print(f"  - {mismatch}")
            return False

        if verbose:
            print(f"✅ 验证通过: {n_checked}条记录全部匹配")
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
        self.top_ks = [10, 100, 1000]
        self.avg_topk_tanimoto = {}
        self.mol_r_map = {}
        self.reward_norm=reward_norm
        self.reward_exp=reward_exp
        self.algo=algo
        self.top1000 = Top1000()
        self.obj_modes_thr_3_4 = ObjectiveModes(obj1_thr=3.4, obj2_thr=0.1, obj3_thr=-4.0)
        self.obj_modes_thr_4_0 = ObjectiveModes(obj1_thr=4.0, obj2_thr=0.1, obj3_thr=-4.0)
        with open("molecule_results_n.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "smiles",
                "reward",
                "core",
                "mol_file_idx",
                "selected_alpha",
                "objective_dim1",
                "objective_dim2",
                "objective_dim3",
                "hit_obj_mode_cond_3_4",
                "hit_obj_mode_cond_4_0",
            ])
        # 初始化分子对象保存文件（使用gzip压缩以减小文件大小）
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mol_file_path = f"molecule_objects_{ts}.pkl.gz"
        self.mol_file = gzip.open(self.mol_file_path, "wb")
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
                self.top1000.add((r,m))

            objective_values = getattr(m, "objective_values", None)
            self.obj_modes_thr_3_4.add(m, objective_values)
            self.obj_modes_thr_4_0.add(m, objective_values)

            selected_alpha = ""
            dim1 = ""
            dim2 = ""
            dim3 = ""
            hit_3_4 = 0
            hit_4_0 = 0
            preference = getattr(m, "preference", None)
            if preference is not None:
                pref_np = np.asarray(preference, dtype=np.float64).reshape(-1)
                if pref_np.shape[0] > 0:
                    selected_alpha = json.dumps(pref_np.tolist())
            if objective_values is not None:
                obj_np = np.asarray(objective_values, dtype=np.float64).reshape(-1)
                if obj_np.shape[0] >= 3:
                    dim1 = float(obj_np[0])
                    dim2 = float(obj_np[1])
                    dim3 = float(obj_np[2])
                    hit_3_4 = int((dim1 > 3.4) and (dim2 > 0.1) and (dim3 > -4.0))
                    hit_4_0 = int((dim1 > 4.0) and (dim2 > 0.1) and (dim3 > -4.0))

            assert action_states[0]-853 >=0 and action_states[0]-853 <32
            self.avg_core[action_states[0]-853] += r
            self.cnt_core[action_states[0]-853] += 1
            with open("molecule_results_n.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    m.smiles,
                    r,
                    action_states[0]-853,
                    self.mol_file_index,
                    selected_alpha,
                    dim1,
                    dim2,
                    dim3,
                    hit_3_4,
                    hit_4_0,
                ])
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
        num_obj_modes_3_4 = self.obj_modes_thr_3_4.get_modes()
        num_obj_modes_4_0 = self.obj_modes_thr_4_0.get_modes()
        print("Average reward per core:", [avg / cnt if cnt > 0 else 0 for avg, cnt in zip(self.avg_core, self.cnt_core)])
        print("Visit time per core:", [cnt for cnt in self.cnt_core])
        return (
            avg_topk_rs,
            avg_topk_tanimoto,
            num_obj_modes_3_4,
            num_obj_modes_4_0,
        )
    
    def verify_csv_pkl_match(self, csv_path="molecule_results_n.csv", pkl_path=None, verbose=True):
        """
        验证CSV文件和pickle文件是否对应（调用独立验证函数）
        
        Args:
            csv_path: CSV文件路径
            pkl_path: pickle文件路径（默认使用本次运行写出的pkl文件）
            verbose: 是否打印详细信息
            
        Returns:
            bool: 如果所有记录都匹配返回True，否则返回False
        """
        if pkl_path is None:
            pkl_path = getattr(self, "mol_file_path", "molecule_objects.pkl.gz")
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify molecule_results CSV matches molecule_objects pickle stream.")
    parser.add_argument(
        "--csv",
        default="molecule_results_n.csv",
        help="CSV path (default: molecule_results_n.csv)",
    )
    parser.add_argument(
        "--pkl",
        default="molecule_objects_20260213_151908.pkl.gz",
        help="Pickle.gz path (default: molecule_objects_20260213_151908.pkl.gz)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print details; rely on exit code only.",
    )
    args = parser.parse_args()
    print(args.pkl,args.csv)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve(p: str) -> str:
        if os.path.exists(p):
            return p
        candidate = os.path.join(script_dir, p)
        if os.path.exists(candidate):
            return candidate
        return p

    csv_path = _resolve(args.csv)
    pkl_path = _resolve(args.pkl)

    ok = verify_csv_pkl_files(csv_path=csv_path, pkl_path=pkl_path, verbose=(not args.quiet))
    sys.exit(0 if ok else 1)