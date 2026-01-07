import json

# 读取 JSON 文件
with open('blocks_sub.json', 'r') as f:
    data = json.load(f)

# 处理 block_r 字段
for key, value in data['block_r'].items():
    if len(value) > 1:
        data['block_r'][key] = [value[0]]

# 保存处理后的数据（可选）
with open('blocks_sub_one_step.json', 'w') as f:
    json.dump(data, f, indent=2)

# 或者直接输出处理后的数据
print(json.dumps(data, indent=2))