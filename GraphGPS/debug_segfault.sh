#!/bin/bash
# 段错误调试脚本

echo "=== 段错误调试工具 ==="
echo ""
echo "选择调试方法:"
echo "1. 使用gdb调试（推荐）"
echo "2. 使用strace跟踪系统调用"
echo "3. 使用faulthandler（已内置）"
echo "4. 禁用多进程运行"
echo "5. 使用CPU模式运行"
echo "6. 设置单线程运行"
echo ""

read -p "请选择 (1-6): " choice

CONFIG_FILE="${1:-configs/GPS/a-mols.yaml}"

case $choice in
    1)
        echo "使用gdb调试..."
        if ! command -v gdb &> /dev/null; then
            echo "错误: gdb未安装，请先安装: apt-get install -y gdb"
            exit 1
        fi
        gdb -ex run -ex bt -ex quit --args python main.py --cfg "$CONFIG_FILE"
        ;;
    2)
        echo "使用strace跟踪系统调用..."
        strace -o trace.log python main.py --cfg "$CONFIG_FILE"
        echo "跟踪结果已保存到 trace.log，查看最后几行:"
        tail -50 trace.log
        ;;
    3)
        echo "使用faulthandler运行..."
        PYTHONFAULTHANDLER=1 python main.py --cfg "$CONFIG_FILE"
        ;;
    4)
        echo "禁用多进程运行..."
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        python main.py --cfg "$CONFIG_FILE"
        ;;
    5)
        echo "使用CPU模式运行..."
        CUDA_VISIBLE_DEVICES="" python main.py --cfg "$CONFIG_FILE"
        ;;
    6)
        echo "设置单线程运行..."
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        python main.py --cfg "$CONFIG_FILE"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac



