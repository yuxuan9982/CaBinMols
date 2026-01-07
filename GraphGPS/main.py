import datetime
import os
import sys
import torch
import logging
import faulthandler

# Enable faulthandler to get stack trace on segfault
faulthandler.enable()
# Also write to stderr
faulthandler.enable(file=sys.stderr, all_threads=True)

import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    try:
        print("[DEBUG] Step 1: Loading command line arguments...", flush=True)
        # Load cmd line args
        args = parse_args()
        print(f"[DEBUG] Step 2: Args loaded: {args}", flush=True)
        
        print("[DEBUG] Step 3: Loading config file...", flush=True)
        # Load config file
        set_cfg(cfg)
        load_cfg(cfg, args)
        custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
        dump_cfg(cfg)
        print("[DEBUG] Step 4: Config loaded and dumped", flush=True)
        
        print("[DEBUG] Step 5: Setting PyTorch environment...", flush=True)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        print(f"[DEBUG] PyTorch threads set to: {cfg.num_threads}", flush=True)
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}", flush=True)
        
        # Repeat for multiple experiment runs
        print("[DEBUG] Step 6: Setting up run loop...", flush=True)
        run_ids, seeds, split_indices = run_loop_settings()
        print(f"[DEBUG] Run IDs: {run_ids}, Seeds: {seeds}, Split indices: {split_indices}", flush=True)
        
        for run_id, seed, split_index in zip(run_ids, seeds, split_indices):
            print(f"[DEBUG] ========== Starting Run ID {run_id} ==========", flush=True)
            # Set configurations for each run
            print("[DEBUG] Step 7: Setting run directory...", flush=True)
            custom_set_run_dir(cfg, run_id)
            print("[DEBUG] Step 8: Setting printing...", flush=True)
            set_printing()
            cfg.dataset.split_index = split_index
            cfg.seed = seed
            cfg.run_id = run_id
            print(f"[DEBUG] Step 9: Seeding everything with seed={cfg.seed}...", flush=True)
            seed_everything(cfg.seed)
            print("[DEBUG] Step 10: Auto selecting device...", flush=True)
            auto_select_device()
            print(f"[DEBUG] Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}", flush=True)
            
            if cfg.pretrained.dir:
                print("[DEBUG] Step 11: Loading pretrained model config...", flush=True)
                cfg = load_pretrained_model_cfg(cfg)
            
            logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                         f"split_index={cfg.dataset.split_index}")
            logging.info(f"    Starting now: {datetime.datetime.now()}")
            
            # Set machine learning pipeline
            print("[DEBUG] Step 12: Creating data loaders...", flush=True)
            loaders = create_loader()
            print(f"[DEBUG] Loaders created: {type(loaders)}", flush=True)
            
            print("[DEBUG] Step 13: Creating loggers...", flush=True)
            loggers = create_logger()
            print(f"[DEBUG] Loggers created: {type(loggers)}", flush=True)
            
            print("[DEBUG] Step 14: Creating model...", flush=True)
            model = create_model()
            print(f"[DEBUG] Model created: {type(model)}", flush=True)
            
            if cfg.pretrained.dir:
                print("[DEBUG] Step 15: Initializing model from pretrained...", flush=True)
                model = init_model_from_pretrained(
                    model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                    cfg.pretrained.reset_prediction_head, seed=cfg.seed
                )
            
            print("[DEBUG] Step 16: Creating optimizer...", flush=True)
            optimizer = create_optimizer(model.parameters(),
                                         new_optimizer_config(cfg))
            print(f"[DEBUG] Optimizer created: {type(optimizer)}", flush=True)
            
            print("[DEBUG] Step 17: Creating scheduler...", flush=True)
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            print(f"[DEBUG] Scheduler created: {type(scheduler)}", flush=True)
            
            # Print model info
            logging.info(model)
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info('Num parameters: %s', cfg.params)
            
            # Start training
            print("[DEBUG] Step 18: Starting training...", flush=True)
            if cfg.train.mode == 'standard':
                if cfg.wandb.use:
                    logging.warning("[W] WandB logging is not supported with the "
                                    "default train.mode, set it to `custom`")
                print("[DEBUG] Using standard training mode...", flush=True)
                datamodule = GraphGymDataModule()
                print("[DEBUG] DataModule created, starting train()...", flush=True)
                train(model, datamodule, logger=True)
            else:
                print(f"[DEBUG] Using custom training mode: {cfg.train.mode}...", flush=True)
                train_func = train_dict[cfg.train.mode]
                print(f"[DEBUG] Training function: {train_func}", flush=True)
                train_func(loggers, loaders, model, optimizer, scheduler)
            print(f"[DEBUG] ========== Run ID {run_id} completed ==========", flush=True)
    except Exception as e:
        print(f"[DEBUG] Exception caught: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
