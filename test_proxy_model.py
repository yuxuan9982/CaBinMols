from GraphGPS.recover_model import load_trained_model,process
import argparse,os
import torch
from torch_geometric.graphgym.config import cfg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    cfg_file = args.cfg
    ckpt = args.ckpt or os.path.join(
        os.path.dirname(cfg_file).replace("configs", "results"),
        os.path.splitext(os.path.basename(cfg_file))[0],
        "0/model_best.pth"
    )
    model = load_trained_model(cfg_file, ckpt)
    print("✅ Model successfully loaded and ready for inference.")

    # batch = process(input("Input Smiles:"))
    batch = process("Cc1cc(C)c(N2CN(c3c(C)cc(C)cc3C)C(=O)C2=O)c(C)c1")
    # batch = add_lap_pe(batch)
    batch = batch.to(cfg.accelerator)
    print(batch)
    with torch.no_grad():
        output = model(batch)
    print(output)
#python test_proxy_model.py --cfg GraphGPS/configs/GPS/a-mols.yaml --ckpt GraphGPS/results/models/model_best.pth
