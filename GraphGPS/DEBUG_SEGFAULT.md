# 段错误调试指南

## 方法1: 使用修改后的main.py（已添加调试输出）

运行修改后的代码，它会输出每个步骤的调试信息，帮助定位段错误发生的位置：

```bash
python main.py --cfg configs/GPS/a-mols.yaml
```

查看最后输出的 `[DEBUG] Step X` 来确定段错误发生在哪个步骤。

## 方法2: 使用gdb调试（推荐）

### 安装gdb（如果还没有）
```bash
apt-get update && apt-get install -y gdb
```

### 使用gdb运行
```bash
gdb --args python main.py --cfg configs/GPS/a-mols.yaml
```

在gdb中：
```
(gdb) run
# 等待段错误发生
(gdb) bt          # 查看堆栈跟踪
(gdb) info registers  # 查看寄存器状态
(gdb) quit
```

### 或者直接运行并自动生成backtrace
```bash
gdb -ex run -ex bt -ex quit --args python main.py --cfg configs/GPS/a-mols.yaml
```

## 方法3: 使用Python的faulthandler（已添加到代码中）

代码已经启用了 `faulthandler`，段错误时会自动输出堆栈跟踪。

如果需要在运行时启用更详细的输出：
```bash
PYTHONFAULTHANDLER=1 python main.py --cfg configs/GPS/a-mols.yaml
```

## 方法4: 使用strace跟踪系统调用

```bash
strace -o trace.log python main.py --cfg configs/GPS/a-mols.yaml
```

然后查看 `trace.log` 文件的最后几行，看看最后调用的系统调用是什么。

## 方法5: 检查常见问题

### 5.1 多进程数据加载问题

如果段错误发生在 `create_loader()` 步骤，可能是多进程数据加载的问题。尝试：

1. 在配置文件中设置 `num_workers: 0`（如果存在）
2. 或者设置环境变量：
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python main.py --cfg configs/GPS/a-mols.yaml
```

### 5.2 CUDA相关问题

如果使用CUDA，尝试：
```bash
CUDA_LAUNCH_BLOCKING=1 python main.py --cfg configs/GPS/a-mols.yaml
```

这会同步CUDA操作，更容易定位问题。

### 5.3 内存问题

检查是否有内存不足：
```bash
dmesg | tail -20
```

### 5.4 检查依赖库版本

段错误可能是由于库版本不兼容：
```bash
pip list | grep -E "torch|geometric|numpy"
```

## 方法6: 逐步注释代码

如果上述方法都无法定位，可以逐步注释 `main.py` 中的代码，找到导致段错误的最小代码块。

## 常见段错误原因

1. **多进程数据加载**: PyTorch DataLoader 的 `num_workers > 0` 在某些环境下会导致段错误
2. **C++扩展库问题**: torch_geometric 或其他C++扩展库的版本不兼容
3. **内存访问错误**: 访问了无效的内存地址
4. **CUDA驱动问题**: CUDA版本与PyTorch版本不匹配
5. **共享内存问题**: 在多进程环境下共享内存不足

## 快速修复尝试

1. **禁用多进程**:
```python
# 在配置文件中或代码中设置
cfg.dataset.num_workers = 0
```

2. **使用CPU模式**:
```bash
CUDA_VISIBLE_DEVICES="" python main.py --cfg configs/GPS/a-mols.yaml
```

3. **设置线程数**:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python main.py --cfg configs/GPS/a-mols.yaml
```



