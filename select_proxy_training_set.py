import argparse
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

try:
    # Newer RDKit API (recommended; avoids deprecation warnings)
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    _HAVE_MORGAN_GENERATOR = True
except Exception:
    # Fallback for older RDKit versions
    GetMorganGenerator = None  # type: ignore
    _HAVE_MORGAN_GENERATOR = False


@dataclass
class Candidate:
    row: Dict[str, str]
    smiles: str
    reward: float
    obj1: float
    obj2: float
    obj3: float
    core: str
    quality: float = 0.0
    fp: Optional[DataStructs.ExplicitBitVect] = None


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = values.mean()
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def _replace_placeholders_with_h(smiles: str) -> Optional[str]:
    """
    Follow process_Htop1w.ipynb:
    - Replace placeholder atoms ('*' or atomicNum==0) with H
    - Sanitize
    - Remove explicit Hs (let RDKit implicitize)
    - Return canonical isomeric SMILES without explicit [H]
    """
    s = (smiles or "").strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None

        rw = Chem.RWMol(mol)
        replace_indices = [
            a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "*" or a.GetAtomicNum() == 0
        ]
        for idx in reversed(replace_indices):
            rw.ReplaceAtom(idx, Chem.Atom(1))  # atomic number 1 = H

        mol2 = rw.GetMol()
        Chem.SanitizeMol(mol2)
        mol_no_h = Chem.RemoveHs(mol2)
        return Chem.MolToSmiles(mol_no_h, isomericSmiles=True)
    except Exception:
        return None


def load_candidates(csv_path: str) -> List[Candidate]:
    # Keep only one row per SMILES (highest reward), avoiding repeated molecules.
    best_by_smiles: Dict[str, Candidate] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"smiles", "reward", "core", "objective_dim1", "objective_dim2", "objective_dim3"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")
        for row in reader:
            raw_smiles = (row.get("smiles") or "").strip()
            if not raw_smiles:
                continue
            smiles_clean = _replace_placeholders_with_h(raw_smiles)
            if not smiles_clean:
                continue

            # Keep the raw SMILES for traceability, but make `smiles` usable downstream.
            row2 = dict(row)
            row2.setdefault("smiles_raw", raw_smiles)
            row2["smiles_clean"] = smiles_clean
            row2["smiles"] = smiles_clean
            cand = Candidate(
                row=row2,
                smiles=smiles_clean,
                reward=_safe_float(row.get("reward", "")),
                obj1=_safe_float(row.get("objective_dim1", "")),
                obj2=_safe_float(row.get("objective_dim2", "")),
                obj3=_safe_float(row.get("objective_dim3", "")),
                core=str(row.get("core", "")),
            )
            prev = best_by_smiles.get(smiles_clean)
            if prev is None or cand.reward > prev.reward:
                best_by_smiles[smiles_clean] = cand
    return list(best_by_smiles.values())


def build_quality(cands: List[Candidate]) -> None:
    rewards = np.asarray([c.reward for c in cands], dtype=np.float64)
    obj1 = np.asarray([c.obj1 for c in cands], dtype=np.float64)
    obj2 = np.asarray([c.obj2 for c in cands], dtype=np.float64)
    obj3 = np.asarray([c.obj3 for c in cands], dtype=np.float64)
    core_counts = Counter(c.core for c in cands)

    r_z = _zscore(rewards)
    o1_z = _zscore(obj1)
    o2_z = _zscore(obj2)
    o3_z = _zscore(obj3)
    novelty = np.asarray([1.0 / math.sqrt(max(1, core_counts[c.core])) for c in cands], dtype=np.float64)
    novelty = _zscore(novelty)

    # Weighted quality:
    # - reward dominates
    # - objective dims keep multi-objective signal
    # - novelty helps avoid over-concentrating on common cores
    quality = 0.55 * r_z + 0.15 * o1_z + 0.10 * o2_z + 0.10 * o3_z + 0.10 * novelty

    for i, c in enumerate(cands):
        c.quality = float(quality[i])


def build_fingerprints(cands: List[Candidate], radius: int = 2, n_bits: int = 2048) -> List[Candidate]:
    valid: List[Candidate] = []
    fpgen = GetMorganGenerator(radius=radius, fpSize=n_bits) if _HAVE_MORGAN_GENERATOR else None
    for c in cands:
        mol = Chem.MolFromSmiles(c.smiles)
        if mol is None:
            continue
        if fpgen is not None:
            c.fp = fpgen.GetFingerprint(mol)
        else:
            # Deprecated in newer RDKit; kept for compatibility only.
            c.fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        valid.append(c)
    return valid


def diverse_select(
    ranked: List[Candidate],
    k: int,
    max_sim: float,
    max_per_core: int,
) -> List[Candidate]:
    selected: List[Candidate] = []
    per_core = Counter()
    thresholds = [max_sim, min(0.85, max_sim + 0.08), 0.92, 0.98, 1.01]

    for thr in thresholds:
        for c in ranked:
            if len(selected) >= k:
                break
            if per_core[c.core] >= max_per_core:
                continue
            if not selected:
                selected.append(c)
                per_core[c.core] += 1
                continue
            sims = DataStructs.BulkTanimotoSimilarity(c.fp, [s.fp for s in selected])
            if sims and max(sims) > thr:
                continue
            selected.append(c)
            per_core[c.core] += 1
        if len(selected) >= k:
            break

    # Fill remainder by quality if constraints were too strict.
    if len(selected) < k:
        chosen_ids = {id(x) for x in selected}
        for c in ranked:
            if len(selected) >= k:
                break
            if id(c) in chosen_ids:
                continue
            selected.append(c)
    return selected[:k]


def _compute_bin_targets(
    k: int,
    n_bins: int,
    beta: float,
    min_per_bin: int,
) -> List[int]:
    """
    Allocate k samples across n_bins with larger weights for higher-reward bins.

    We use exponentially increasing weights: w_i = exp(beta * i/(n_bins-1)).
    This keeps coverage across the full reward range while mildly emphasizing
    higher-reward regions (useful for proxy accuracy in optimization settings).
    """
    if n_bins <= 0:
        raise ValueError("--reward-bins 必须为正整数")
    if k <= 0:
        return [0] * n_bins

    weights = np.exp(beta * np.linspace(0.0, 1.0, n_bins, dtype=np.float64))
    weights_sum = float(weights.sum())
    raw = weights / max(1e-12, weights_sum) * float(k)
    targets = [int(math.floor(x)) for x in raw.tolist()]

    # Enforce minimum per bin if possible.
    if min_per_bin > 0:
        need = 0
        for i in range(n_bins):
            if targets[i] < min_per_bin:
                need += (min_per_bin - targets[i])
                targets[i] = min_per_bin
        # If we overshot k, take back from high bins first (still keeps coverage).
        if sum(targets) > k:
            overshoot = sum(targets) - k
            for i in range(n_bins - 1, -1, -1):
                if overshoot <= 0:
                    break
                take = min(overshoot, max(0, targets[i] - min_per_bin))
                targets[i] -= take
                overshoot -= take

    # Distribute remaining counts to higher bins first.
    remaining = k - sum(targets)
    for i in range(n_bins - 1, -1, -1):
        if remaining <= 0:
            break
        targets[i] += 1
        remaining -= 1
        if i == 0 and remaining > 0:
            # Wrap around if k is very large; keep adding to high bins.
            i = n_bins - 1

    # Final guard: make sure sum is exactly k.
    total = sum(targets)
    if total > k:
        # Remove extras from the lowest bins first.
        extra = total - k
        for i in range(n_bins):
            if extra <= 0:
                break
            take = min(extra, targets[i])
            targets[i] -= take
            extra -= take
    elif total < k:
        missing = k - total
        targets[-1] += missing

    return targets


def _assign_reward_bins_by_rank(cands: List[Candidate], n_bins: int) -> List[List[Candidate]]:
    """
    Rank-based binning by reward (stable under heavy reward ties).
    Each bin gets roughly equal number of candidates.
    """
    if n_bins <= 1:
        return [list(cands)]
    ordered = sorted(cands, key=lambda x: x.reward)
    bins: List[List[Candidate]] = [[] for _ in range(n_bins)]
    n = len(ordered)
    for idx, c in enumerate(ordered):
        b = int(idx * n_bins / max(1, n))
        if b >= n_bins:
            b = n_bins - 1
        bins[b].append(c)
    return bins


def stratified_reward_select(
    cands: List[Candidate],
    k: int,
    max_sim: float,
    max_per_core: int,
    reward_bins: int,
    reward_bin_beta: float,
    min_per_bin: int,
) -> List[Candidate]:
    """
    Selection tuned for proxy regression accuracy:
    - cover the full reward distribution via stratification
    - within each reward stratum, prefer higher-quality samples
    - maintain global diversity constraints (similarity threshold + core cap)
    """
    if k <= 0:
        return []
    if not cands:
        return []

    thresholds = [max_sim, min(0.85, max_sim + 0.08), 0.92, 0.98, 1.01]
    bins = _assign_reward_bins_by_rank(cands, reward_bins)

    # Within each bin, shuffle to avoid ordering bias then sort by quality.
    for b in bins:
        random.shuffle(b)
        b.sort(key=lambda x: x.quality, reverse=True)

    n_bins = len(bins)
    if k < n_bins:
        # If k is tiny, avoid forcing coverage.
        min_per_bin = 0

    targets = _compute_bin_targets(k=k, n_bins=n_bins, beta=reward_bin_beta, min_per_bin=min_per_bin)
    selected: List[Candidate] = []
    per_core: Counter = Counter()
    chosen_ids = set()
    selected_per_bin = [0] * n_bins
    ptr = [0] * n_bins

    # Prefer filling higher-reward bins first (still keeping lower-bin quotas via targets).
    bin_order = list(range(n_bins - 1, -1, -1))

    for thr in thresholds:
        progressed = True
        while progressed and len(selected) < k:
            progressed = False
            for bi in bin_order:
                if len(selected) >= k:
                    break
                if selected_per_bin[bi] >= targets[bi]:
                    continue
                b = bins[bi]
                while ptr[bi] < len(b) and selected_per_bin[bi] < targets[bi] and len(selected) < k:
                    c = b[ptr[bi]]
                    ptr[bi] += 1
                    cid = id(c)
                    if cid in chosen_ids:
                        continue
                    if per_core[c.core] >= max_per_core:
                        continue
                    if selected:
                        sims = DataStructs.BulkTanimotoSimilarity(c.fp, [s.fp for s in selected])
                        if sims and max(sims) > thr:
                            continue
                    selected.append(c)
                    chosen_ids.add(cid)
                    per_core[c.core] += 1
                    selected_per_bin[bi] += 1
                    progressed = True
                    break

    # If still not enough, fill by global quality while keeping (soft) constraints first.
    if len(selected) < k:
        ranked = list(cands)
        random.shuffle(ranked)
        ranked.sort(key=lambda x: x.quality, reverse=True)

        # First: keep core cap; similarity effectively disabled at 1.01.
        for c in ranked:
            if len(selected) >= k:
                break
            cid = id(c)
            if cid in chosen_ids:
                continue
            if per_core[c.core] >= max_per_core:
                continue
            selected.append(c)
            chosen_ids.add(cid)
            per_core[c.core] += 1

        # Final fallback: ignore core cap too (guarantee k).
        if len(selected) < k:
            for c in ranked:
                if len(selected) >= k:
                    break
                cid = id(c)
                if cid in chosen_ids:
                    continue
                selected.append(c)
                chosen_ids.add(cid)

    return selected[:k]


def write_output(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 molecule_results_n.csv 中选择 K 个更适合 proxy model 训练的分子子集。"
    )
    parser.add_argument("--input", default="molecule_results_n.csv", help="输入 CSV 路径")
    parser.add_argument("--k", type=int, required=True, help="要选择的分子数量")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 路径（默认: selected_for_proxy_k{K}.csv）",
    )
    parser.add_argument(
        "--max-sim",
        type=float,
        default=0.72,
        help="多样性约束：与已选分子的最大 Tanimoto 相似度阈值（默认 0.72）",
    )
    parser.add_argument(
        "--max-core-frac",
        type=float,
        default=0.20,
        help="单个 core 占比上限（默认 0.20）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于稳定排序打散）",
    )
    parser.add_argument(
        "--mode",
        choices=["quality_diverse", "stratified_reward"],
        default="stratified_reward",
        help=(
            "选择策略：quality_diverse=按quality高分优先+多样性；"
            "stratified_reward=按reward分层覆盖（更利于proxy回归准确性）"
        ),
    )
    parser.add_argument(
        "--reward-bins",
        type=int,
        default=10,
        help="reward 分层桶数（仅 stratified_reward 生效，默认 10）",
    )
    parser.add_argument(
        "--reward-bin-beta",
        type=float,
        default=1.2,
        help="分层配额对高 reward 的偏置强度（越大越偏高分；默认 1.2）",
    )
    parser.add_argument(
        "--min-per-bin",
        type=int,
        default=1,
        help="每个 reward 分层至少选择数量（k>=bins 时生效；默认 1）",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k 必须为正整数")
    if not (0.0 < args.max_core_frac <= 1.0):
        raise ValueError("--max-core-frac 必须在 (0, 1] 内")
    if args.reward_bins <= 0:
        raise ValueError("--reward-bins 必须为正整数")
    if args.min_per_bin < 0:
        raise ValueError("--min-per-bin 不能为负数")
    if args.output is None:
        args.output = f"selected_for_proxy_k{args.k}.csv"

    random.seed(args.seed)

    cands = load_candidates(args.input)
    if not cands:
        raise RuntimeError("输入 CSV 没有可用分子")
    if args.k > len(cands):
        raise ValueError(f"--k={args.k} 大于可用唯一分子数 {len(cands)}")

    build_quality(cands)
    cands = build_fingerprints(cands)
    if args.k > len(cands):
        raise ValueError(f"有效 SMILES 分子数不足：请求 {args.k}，实际 {len(cands)}")

    # Shuffle before sorting to make tie-breaking stable but not biased by file order.
    random.shuffle(cands)
    ranked = sorted(cands, key=lambda x: x.quality, reverse=True)
    max_per_core = max(1, int(math.ceil(args.k * args.max_core_frac)))
    if args.mode == "quality_diverse":
        selected = diverse_select(ranked, args.k, max_sim=args.max_sim, max_per_core=max_per_core)
    else:
        selected = stratified_reward_select(
            cands=cands,
            k=args.k,
            max_sim=args.max_sim,
            max_per_core=max_per_core,
            reward_bins=args.reward_bins,
            reward_bin_beta=args.reward_bin_beta,
            min_per_bin=args.min_per_bin,
        )

    fieldnames = list(selected[0].row.keys())
    rows = [c.row for c in selected]
    write_output(args.output, rows, fieldnames)

    top_reward = np.mean([c.reward for c in selected])
    core_cov = len(set(c.core for c in selected))
    print(f"已选择 {len(selected)} 个分子 -> {args.output}")
    print(f"选中集合平均 reward: {top_reward:.4f}")
    print(f"选中集合 core 覆盖数: {core_cov}")
    print(f"参数: max_sim={args.max_sim}, max_core_frac={args.max_core_frac}, seed={args.seed}")


if __name__ == "__main__":
    main()
