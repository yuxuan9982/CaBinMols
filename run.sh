nohup python -u gflownet.py > output_one_step.log 2>&1 &
nohup python -u gflownet.py --ckpt GraphGPS/results/models/model_best_v2_merged.pth > output_one_step_v2_merged.log 2>&1 &