original_files:原始数据
processed_files：处理后的数据
    - process_core 对core进行处理，得到类似gfn的blocks_PDB_105的结构
GraphGPS:
    - python main.py --cfg configs/GPS/a-mols.yaml训练代理模型
    - python recover_model.py --cfg configs/GPS/a-mols.yaml --ckpt results/models/model_best.pth训好的代理模型测试

MDP设计：
    - 第一步一定是加骨架，并且只能加一个骨架
    - 如果一个取代基有多个位点，那就需要建立多个blocks（每个位点对应一种情况，因为连接位点不同对应不同的状态）（gfn默认连接的是新block的第一个位点，所以要么有对称位点，要么就得多种情况）
    - 考虑处理取代基的两个位点和骨架上邻近的两个点拼起来的情况
        · 多加一个combine操作，限定为，两个不属于同一个block的位点拼起来
    
    - 找parents怎么找
        · 怎么判断是否是combine操作？
            · stem数量就可以判断？并且一定只有两种情况
        · 是否会有A+B = C+D这种情况？
            · 应该不会？
        · 一般情况
            先特判骨架，再


todo：
    找parents逻辑修改 √
    第一步添加逻辑修改 √
    添加combine操作，combine的parents怎么定义？
    先不管combine，把流程跑通吧
    

install:
    torch 2.8.0+cu128
    pyg 对应官网
    conda install openbabel fsspec rdkit -c conda-forge

    pip install pytorch-lightning yacs torchmetrics
    pip install performer-pytorch
    pip install tensorboardX
    pip install ogb
    pip install wandb



nohup python gflownet.py > output_one_step_newtest.py 2>&1 &



更新proxy model_best

在Graphgps/graphgps/loader/dataset/smiles_dataset.py里，重新处理文件


nohup python -u gflownet.py --ckpt GraphGPS/results/models/model_best_v2_merged.pth > output_one_step_v2_merged.log 2>&1 &