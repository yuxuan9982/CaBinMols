#!/usr/bin/env python3
"""
将v_i的数据格式转化成v_{i-1}的格式并合并生成v_i_merged版本
"""

import pandas as pd
import argparse
import os

def convert_vi_to_vi_minus_1(vi_df, i):
    """
    将v_i格式转换为v_{i-1}格式
    
    参数:
        vi_df: v_i的DataFrame
        i: 版本号
    
    返回:
        转换后的DataFrame（v_{i-1}格式）
    """
    # v2 -> v1的转换规则
    # v1格式：ID, Class, SMILES, vbur_ratio_vbur_vtot, E_singlet, Energy_triplet, dE_AuCl, dE_triplet
    # v2格式：id, HF_s, HF_tri, dE_triplet, carbene_idx, au_idx, cl_idx, SMILES
    converted_df = pd.DataFrame({
        'ID': vi_df['id'].astype(str),
        'Class': pd.NA,  # v2中没有Class，使用None
        'SMILES': vi_df['SMILES'],
        'vbur_ratio_vbur_vtot': pd.NA,  # v2中没有，使用None
        'E_singlet': vi_df['HF_s'],
        'Energy_triplet': vi_df['HF_tri'],
        'dE_AuCl': pd.NA,  # v2中没有，使用None
        'dE_triplet': vi_df['dE_triplet']
    })
    return converted_df

def main():
    parser = argparse.ArgumentParser(description='将v_i的数据格式转化成v_{i-1}的格式并合并')
    parser.add_argument('-i', '--version', type=int, default=2, 
                       help='版本号i（默认值：2，即v2转换为v1）')
    
    args = parser.parse_args()
    i = args.version
    
    print(f"参数i = {i}")
    print(f"正在将v{i}格式转换为v{i-1}格式...")
    
    # 文件路径
    base_dir = './datasets/custom'
    vi_path = os.path.join(base_dir, f'NHC-cracker-zzy-v{i}.csv')
    vi_minus_1_path = os.path.join(base_dir, f'NHC-cracker-zzy-v{i-1}.csv')
    output_path = os.path.join(base_dir, f'NHC-cracker-zzy-v{i}_merged.csv')
    
    # 检查文件是否存在
    if not os.path.exists(vi_path):
        print(f"错误：文件 {vi_path} 不存在！")
        return
    
    if not os.path.exists(vi_minus_1_path):
        print(f"错误：文件 {vi_minus_1_path} 不存在！")
        return
    
    print(f"正在读取v{i-1}文件: {vi_minus_1_path}")
    vi_minus_1_df = pd.read_csv(vi_minus_1_path)
    print(f"v{i-1}文件包含 {len(vi_minus_1_df)} 行数据")
    
    print(f"正在读取v{i}文件: {vi_path}")
    vi_df = pd.read_csv(vi_path)
    print(f"v{i}文件包含 {len(vi_df)} 行数据")
    
    # 打印v{i-1}的列格式
    print(f"\nv{i-1}的列格式: {', '.join(vi_minus_1_df.columns.tolist())}")
    # 打印v{i}的列格式
    print(f"v{i}的列格式: {', '.join(vi_df.columns.tolist())}")
    
    # 转换v_i格式到v_{i-1}格式
    print(f"\n正在转换v{i}数据格式到v{i-1}格式...")
    vi_converted = convert_vi_to_vi_minus_1(vi_df, i)
    print(f"v{i}转换后包含 {len(vi_converted)} 行数据")
    
    # 合并v_{i-1}和转换后的v_i
    print(f"正在合并v{i-1}和转换后的v{i}数据...")
    merged_df = pd.concat([vi_minus_1_df, vi_converted], ignore_index=True)
    print(f"合并后总共包含 {len(merged_df)} 行数据")
    
    # 保存合并后的文件
    print(f"正在保存到 {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print("完成！")
    
    # 显示前几行数据作为验证
    print(f"\n合并后的数据预览（前5行）：")
    print(merged_df.head())
    
    print(f"\n合并后的数据统计：")
    print(f"- 总行数: {len(merged_df)}")
    print(f"- 列数: {len(merged_df.columns)}")
    print(f"- 列名: {list(merged_df.columns)}")
    
    # 打印转换信息
    print(f"\n转换信息：")
    print(f"- 源文件: v{i} ({len(vi_df)} 行)")
    print(f"- 目标格式: v{i-1} ({len(vi_minus_1_df)} 行)")
    print(f"- 转换后: {len(vi_converted)} 行")
    print(f"- 合并后: {len(merged_df)} 行")
    print(f"- 输出文件: {output_path}")

if __name__ == '__main__':
    main()
