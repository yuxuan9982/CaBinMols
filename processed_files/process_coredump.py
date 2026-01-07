import json
from rdkit import Chem

# === 输入 SMILES 数据 ===
nhc_core_smiles_subidx = {
    'A': '[*:1]N1CN([*:2])C([*:3])=C1[*:4]',
    'B': '[*:1]N1CN([*:2])C([*:3])=N1',
    'C': '[*:1]N1CN([*:2])C2=C1C([*:3])=C([*:4])C([*:5])=C2[*:6]',
    'D': '[*:1]N1CN([*:2])C([*:3])([*:4])C1([*:5])[*:6]',
    'E1': '[*:1]N1C([*:2])([*:3])C([*:4])([*:5])C([*:6])([*:7])N([*:8])C1',
    'E2': '[*:1]N1CN([*:2])C([*:3])([*:4])C([*:5])([*:6])C([*:7])([*:8])C1([*:9])[*:10]',
    'E3': '[*:1]N1CN([*:2])C([*:3])([*:4])C([*:5])([*:6])C([*:7])([*:8])C([*:9])([*:10])C1([*:11])[*:12]',
    'H': '[*:1]N1CC([*:2])=C([*:3])N1[*:4]',
    'I': '[*:1]N(N1[*:2])CC2=C1C([*:3])=C([*:4])C([*:5])=C2[*:6]',
    'J1': '[*:1]N1COC([*:2])([*:3])C1([*:4])[*:5]',
    'J2': '[*:1]N1COc2c([*:2])c([*:3])c([*:4])c([*:5])c12',
    'J3': '[*:1]N1C([*:2])=C([*:3])OC1',
    'L1': '[*:1]n1c([*:2])nc([*:3])c1',
    'M1': '[*:1]N1CSC([*:2])=C1[*:3]',
    'M2': '[*:1]N1c2c([*:2])c([*:3])c([*:4])c([*:5])c2SC1',
    'N': '[*:1]P([*:2])([*:3])=C1C([*:4])=C([*:5])N([*:6])C1',
    'O': '[*:1]N1CN([*:2])C2=C3C1=C([*:3])C([*:4])=C([*:5])C3=C([*:6])C([*:7])=C2[*:8]',
    'P1': 'O=C1N([*:1])CN([*:2])C([*:3])([*:4])C1([*:5])[*:6]',
    'P2': 'O=C1N([*:1])CN([*:2])C([*:3])=C1[*:4]',
    'Q1': '[*:1]N1C([*:2])([*:3])C([*:4])([*:5])C([*:6])([*:7])C1',
    'Q2': '[*:1]N1CC([*:2])([*:3])C([*:4])([*:5])C([*:6])([*:7])C1([*:8])[*:9]',
    'Q3': '[*:1]N1c2c([*:2])c([*:3])c([*:4])c([*:5])c2C([*:6])([*:7])C1',
    'S': '[*:1]N([*:2])C(C1)=C1N([*:3])[*:4]',
    'new_1': '[*:1]N1c2nc([*:2])c([*:3])nc2N([*:4])C1',
    'new_2': '[*:1]N1c2c([*:2])c([*:3])nc([*:4])c2N([*:5])C1',
    'new_3': '[*:1]N1CN([*:2])c(c1c(=O)n2[*:3])n([*:4])c2=O',
    'new_4': 'O=C1N([*:1])CN([*:2])N1[*:3]',
    'new_5': 'O=c1n([*:1])c([*:2])nc2c1N([*:3])CN2[*:4]',
    'new_6': '[*:1]N1c2c([*:2])c([*:3])c([*:4])nc2N([*:5])C1',
    'new_7': '[*:1]N1c2nc([*:2])nc([*:3])c2N([*:4])C1',
    'new_8': '[*:1]N1c2c([*:2])sc([*:3])c2N([*:4])C1'
}

nhc_core_smiles_with_ring = {
    'A': 'C1NC=CN1', 'B': 'C1NN=CN1', 'C': 'C1(C=CC=C2)=C2NCN1',
    'D': 'C1NCCN1', 'E1': 'N1CCCNC1', 'E2': 'C1NCCCCN1',
    'E3': 'C1NCCCCCN1', 'H': 'C1NNC=C1', 'I': 'C1(C=CC=C2)=C2CNN1',
    'J1': 'C1NCCO1', 'J2': 'c12ccccc1OCN2', 'J3': 'N1C=COC1',
    'L1': 'c1cnc[nH]1', 'M1': 'C1NC=CS1', 'M2': 'c12c(NCS2)cccc1',
    'N': '[H]P([H])([H])=C1C=CNC1', 'O': 'C1(NCN2)=C3C2=CC=CC3=CC=C1',
    'P1': 'O=C1NCNCC1', 'P2': 'O=C1NCNC=C1', 'Q1': 'C1NCCC1',
    'Q2': 'N1CCCCC1', 'Q3': 'c12c(NCC2)cccc1', 'S': 'NC1=C(N)C1',
    'new_1': 'c12c(NCN2)nccn1', 'new_2': 'c12c(NCN2)ccnc1',
    'new_3': 'CN1CN(c(c1c(=O)n2C)n(C)c2=O)C', 'new_4': 'O=C1NCNN1',
    'new_5': 'O=c1c(NCN2)c2nc[nH]1', 'new_6': 'c12c(NCN2)cccn1',
    'new_7': 'c12c(NCN2)ncnc1', 'new_8': 'c12c(NCN2)csc1'
}

# === 提取函数 ===
# def extract_attachment_points(smiles):
#     """解析 SMILES 并返回 (纯SMILES, 反应位点索引列表)"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None, []
#     # 找出所有 Dummy 原子（原子符号为 '*'）
#     attach_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
#     # 生成去掉 [*] 的“干净”SMILES
#     mol_no_attach = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[*]'))
#     clean_smiles = Chem.MolToSmiles(mol_no_attach, canonical=True)
#     return attach_atoms

# def normalize_with_attachment(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError("Invalid SMILES")

#     # Step 1. 记录每个 [*] 连接的真实原子索引
#     attach_map = []
#     for atom in mol.GetAtoms():
#         if atom.GetSymbol() == "*":
#             nbrs = atom.GetNeighbors()
#             if len(nbrs) == 1:
#                 attach_map.append(nbrs[0].GetIdx())
#             else:
#                 attach_map.append(None)  # 孤立 [*]，理论上不应该出现

#     # Step 2. 删除所有 [*]
#     to_remove = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
#     em = Chem.EditableMol(mol)
#     for idx in sorted(to_remove, reverse=True):
#         em.RemoveAtom(idx)
#     mol_clean = em.GetMol()
#     Chem.SanitizeMol(mol_clean)

#     # Step 3. 生成 canonical SMILES 并得到原子重编号映射
#     smiles_canonical = Chem.MolToSmiles(mol_clean, canonical=True)
#     mol_canonical = Chem.MolFromSmiles(smiles_canonical)
#     mapping = mol_clean.GetPropsAsDict(includePrivate=True).get('_smilesAtomOutputOrder')
#     # mapping: canonical 序中的原子在原分子中的索引顺序

#     # Step 4. 反向映射，得到新的 attachment 位点编号
#     new_attach = []
#     if mapping:
#         old2new = {old: new for new, old in enumerate(mapping)}
#         for old_idx in attach_map:
#             new_attach.append(old2new.get(old_idx, None))
#     else:
#         new_attach = attach_map  # 若 RDKit 版本不返回映射

#     return smiles_canonical, new_attach

# # === 构建映射 ===
# block_smi = {}
# block_r = {}

# for name, s in nhc_core_smiles_subidx.items():
#     smiles_canonical, attaches = normalize_with_attachment(s)
#     block_smi[name] = nhc_core_smiles_with_ring.get(name, "")
#     print(smiles_canonical,block_smi[name] )
#     assert smiles_canonical == Chem.MolToSmiles(Chem.MolFromSmiles(block_smi[name]))
#     block_r[name] = attaches

# # === 合并输出结构 ===
# output_data = {
#     "block_smi": block_smi,
#     "block_r": block_r
# }

# # === 写入 JSON 文件 ===
# with open("core_blocks_compact.json", "w", encoding="utf-8") as f:
#     json.dump(output_data, f, ensure_ascii=False, indent=4)

# print("✅ 已生成 core_blocks_compact.json。")

print(Chem.MolToSmiles(Chem.MolFromSmiles('[*:1]N1CN([*:2])C([*:3])=C1[*:4]')) )