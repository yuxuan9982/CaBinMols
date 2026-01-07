import pandas as pd
import re
import json
from rdkit import Chem

def add_consecutive_labels(smiles: str) -> str:
    """为 SMILES 中的未编号 [*] 添加连续编号 [*:1], [*:2], ..."""
    pattern = r"\[\*\](?!:)"  # 匹配未编号的 [*]
    count = 0
    def repl(_):
        nonlocal count
        count += 1
        return f"[*:{count}]"
    return re.sub(pattern, repl, smiles)

def get_attachment_indices(smiles: str):
    """提取所有连接位点编号（[*:n]）"""
    return [int(x) for x in re.findall(r"\[\*\:(\d+)\]", smiles)]

def swap_labels(smiles: str, a: int, b: int):
    """交换两个连接位点编号，返回新 SMILES"""
    tmp = re.sub(rf"\[\*\:{a}\]", "[*:X]", smiles)
    tmp = re.sub(rf"\[\*\:{b}\]", f"[*:{a}]", tmp)
    tmp = re.sub(r"\[\*\:X\]", f"[*:{b}]", tmp)
    return tmp

def are_sites_symmetric(smiles: str, site1: int, site2: int) -> bool:
    """
    判断两个连接位点是否完全对称。
    交换它们的编号后，如果 canonical SMILES 相同，则认为对称。
    """
    mol1 = Chem.MolFromSmiles(smiles)
    swapped = swap_labels(smiles, site1, site2)
    mol2 = Chem.MolFromSmiles(swapped)
    if mol1 is None or mol2 is None:
        return False
    s1 = Chem.MolToSmiles(mol1, canonical=True)
    s2 = Chem.MolToSmiles(mol2, canonical=True)
    return s1 == s2

# 1. 读取 CSV
df = pd.read_csv("NHCs_sub_template.csv")

# 2. 删除 Frequency 列（如果存在）
if "Frequency" in df.columns:
    df = df.drop(columns=["Frequency"])

# 3. 找到 SMILES 列
smiles_col = None
for col in df.columns:
    if 'smiles' in col.lower():
        smiles_col = col
        break

if smiles_col is None:
    raise ValueError("未找到包含 SMILES 的列，请确认列名是否正确（例如 'smiles'）")

# 4. 自动编号
df[smiles_col] = df[smiles_col].apply(add_consecutive_labels)

# 5. 筛除 [Au] 和 连接位点 >2
valid_rows = []
for _, row in df.iterrows():
    smi = row[smiles_col]
    if "[Au]" in smi:
        print(smi)
        continue
    attach = get_attachment_indices(smi)
    if len(attach) > 2 or len(attach) == 0:
        print(smi)
        continue
    valid_rows.append((smi, attach))

# 6. 处理对称性并生成额外条目
final_rows = []
for smi, attach in valid_rows:
    final_rows.append((smi, attach))
    if len(attach) == 2:
        if not are_sites_symmetric(smi, attach[0], attach[1]):
            # 如果不对称，则添加反向版本
            final_rows.append((smi, attach[::-1]))

# 7. 构建 JSON
block_smi = {str(i): smi for i, (smi, _) in enumerate(final_rows)}
block_r = {str(i): attach for i, (_, attach) in enumerate(final_rows)}

output_data = {"block_smi": block_smi, "block_r": block_r}

# 8. 保存为 JSON
with open("blocks_sub.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ 共保留 {len(final_rows)} 条数据（包含对称性扩增），结果保存至 blocks_sub.json")
