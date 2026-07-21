import os
import random
import argparse
import subprocess
from pathlib import Path
import time
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn

from utils.BRIDGE import BRIDGE
from utils.train_loop import validate, fit_bridge
from utils.utils import param_num, resolve_dynamic_model_name
from utils.data_pipeline import build_split_loaders


def log_print(text, color=None, on_color=None, attrs=None):
    """
    Print a message to the console with optional color formatting.

    This utility function attempts to use third-party libraries (`termcolor` and `pycrayon`) to produce colored or styled console output.
    If these libraries are not available, it gracefully falls back to standard `print` without formatting.

    Parameters
    ----------
    text : str
        The message to be printed to the console.
    color : str, optional
        Text color name supported by `termcolor` (e.g., 'red', 'green').
        If None, the default terminal color is used.
    on_color : str, optional
        Background color name supported by `termcolor`
        (e.g., 'on_blue', 'on_yellow').
    attrs : list of str, optional
        List of text attributes supported by `termcolor`,
        such as ['bold', 'underline'].
    """

    # Attempt to import termcolor for colored terminal output
    try:
        from termcolor import cprint
    except ImportError:
        cprint = None
        
    # Attempt to import pycrayon (optional; not required for basic printing)
    try:
        from pycrayon import CrayonClient
    except ImportError:
        CrayonClient = None
        
    # Use colored printing if available; otherwise fall back to plain print
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.


def _prepare_run_dirs(args, file_name: str):
    """
    Prepare result directories and return (run_name, logs_dir, model_dir, metrics_dir).

    Directory policy:
    - Root defaults to ./results, override via --results_dir
    - logs   -> {results_dir}/logs
    - metrics-> {results_dir}/metrics
    - model  -> args.model_save_path if provided and not default-empty, else {results_dir}/model
    """
    results_dir = Path(getattr(args, "results_dir", "./results"))
    logs_dir = results_dir/"logs"
    metrics_dir = results_dir/"metrics"

    # If user passed --model_save_path, prefer it; otherwise use results_dir/model
    model_dir = Path(getattr(args, "model_save_path", "")) if getattr(args, "model_save_path", "") else (results_dir / "model")

    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{file_name}_{run_id}"
    return run_name, logs_dir, model_dir, metrics_dir


def main(args):
    """
    Main entry point for training the BRIDGE model.

    This function orchestrates the full training pipeline, including:
        - random seed initialization
        - device (CPU/GPU) configuration
        - data loading and preprocessing
        - sequence embedding extraction using a pretrained transformer
        - construction of structural, biochemical, and motif prior features
        - dataset splitting and DataLoader creation
        - model training, validation, learning-rate scheduling, and early stopping
        - model checkpointing and performance reporting
        - persistent logging/config/metrics under results/{logs,model,metrics}

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments specifying runtime configuration.
        Expected attributes include (but are not limited to):
            - seed : int
                Random seed for reproducibility.
            - use_cpu : bool
                Whether to force CPU execution.
            - device_num : int
                GPU device index to use when CUDA is available.
            - train : bool
                Whether to run the training procedure.
            - data_file : str
                Dataset identifier used to locate input files.
            - data_path : str
                Root directory containing input FASTA and feature files.
            - Transformer_path : str
                Path to the pretrained RBPformer model.
            - lr : float
                Initial learning rate for the optimizer.
            - early_stopping : int
                Number of epochs without improvement before early stopping.
    """

    # Fix random seeds for reproducibility across runs
    fix_seed(args.seed)
    
    # Select computation device (CPU or specific CUDA device)
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")
        
    # Explicitly set the CUDA device if GPU is used
    if device.type == 'cuda':
        torch.cuda.set_device(args.device_num)

    # Maximum sequence length used for padding/truncation
    max_length = 101
    
    # Dataset identifier and base data directory
    file_name = args.data_file
    data_path = args.data_path
    Transformer_batch_size = args.batch_size

    if args.train:
        # Start timing the full training procedure
        start_time = time.time()

        # prepare run dirs + open logfile
        run_name, logs_dir, model_dir, metrics_dir = _prepare_run_dirs(args, file_name)
        log_path = logs_dir / f"{run_name}.log"
        log_fp = open(log_path, "a", encoding="utf-8")

        def log_both(msg: str, color=None, attrs=None):
            # write to file
            log_fp.write(msg + "\n")
            log_fp.flush()
            # print to console
            log_print(msg, color=color, attrs=attrs)

        # Write config file (args + key hyperparams)
        config = {
            "run_name": run_name,
            "data_file": args.data_file,
            "data_path": args.data_path,
            "Transformer_path": args.Transformer_path,
            "seed": args.seed,
            "use_cpu": bool(args.use_cpu),
            "device": str(device),
            "device_num": int(args.device_num),
            "train": bool(args.train),
            "validate": bool(getattr(args, "validate", False)),
            "dynamic_predict": bool(getattr(args, "dynamic_predict", False)),
            "max_length": int(max_length),
            "lr_cli": float(args.lr),
            "early_stopping": int(args.early_stopping)
        }


        # Build train/val/test loaders (features computed once via the shared pipeline).
        # Early stopping + checkpoint selection watch the validation set; the sealed test
        # split is left untouched here and reserved for --validate / --dynamic_predict.
        train_loader, val_loader, _ = build_split_loaders(
            data_file=file_name,
            data_path=data_path,
            transformer_path=args.Transformer_path,
            device=device,
            seed=args.seed,
            max_length=max_length,
            transformer_batch_size=Transformer_batch_size,
        )
        # Initialize the BRIDGE model
        model = BRIDGE().to(device)
        
        # Binary classification loss with class imbalance compensation
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
        
        # Adam optimizer with weight decay regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
        
        # Learning rate scheduling parameters
        initial_lrate = 0.0016
        drop = 0.8
        epochs_drop = 5.0
        warmup_epochs = 40

        # include schedule/loss/optimizer info in config
        config.update(
            {
                "lr_schedule": {
                    "initial_lrate": float(initial_lrate),
                    "drop": float(drop),
                    "epochs_drop": float(epochs_drop),
                    "warmup_epochs": int(warmup_epochs),
                }
            }
        )

        config_path = logs_dir / f"{run_name}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        log_both(f"[RUN] {run_name}", color="green", attrs=["bold"])
        log_both(f"[DIR] logs={logs_dir} model={model_dir} metrics={metrics_dir}")
        log_both(f"[CFG] {config_path}")
        
        # Print total number of model parameters
        param_num(model)

        # Train with the shared loop: warm-up + step-decay LR, val-AUC checkpoint selection,
        # and early stopping. The best checkpoint is saved to {model_dir}/{run_name}.pth.
        best = fit_bridge(
            model, device, train_loader, val_loader, criterion, optimizer,
            max_epochs=200,
            warmup_epochs=warmup_epochs,
            initial_lrate=initial_lrate,
            drop=drop,
            epochs_drop=epochs_drop,
            early_stopping=args.early_stopping,
            ckpt_path=model_dir / f"{run_name}.pth",
            log_fn=lambda m: log_both(m, color="green", attrs=["bold"]),
            tag=file_name,
        )
        best_auc = best["best_val_auc"]
        best_acc = best["best_val_acc"]
        best_mcc = best["best_val_mcc"]
        best_prc = best["best_val_prc"]
        best_epoch = best["best_epoch"]

        # Report best validation performance
        # print("{} auc: {:.4f} acc: {:.4f} prc: {:.4f} mcc: {:.4f}".format(file_name, best_auc, best_acc, best_prc, best_mcc))
        summary_line = (
            f"{file_name} best: auc={best_auc:.4f} acc={best_acc:.4f} prc={best_prc:.4f} mcc={best_mcc:.4f} "
            f"(epoch={best_epoch})"
        )
        log_both(summary_line, color="green", attrs=["bold"])
        
        best_summary = {
            "run_name": run_name,
            "data_file": file_name,
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_auc),
            "best_val_acc": float(best_acc),
            "best_val_prc": float(best_prc),
            "best_val_mcc": float(best_mcc),
            "seed": int(args.seed),
            "device": str(device),
            "checkpoint": str((model_dir / f"{run_name}.pth").resolve()),
            "log_file": str(log_path.resolve()),
            "config_file": str(config_path.resolve()),
        }
        best_path = metrics_dir / f"{run_name}_best.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)
        log_both(f"[BEST] {best_path}")

        log_fp.close()

        # Report total training time
        end_time = time.time()
        time_cost = end_time - start_time
        print("Time cost: {:.2f} min".format(time_cost / 60))


    if args.validate:
        """
        Run evaluation on the test split using a previously trained model.

        This mode reloads a saved BRIDGE checkpoint and evaluates it on
        the held-out test set constructed from the same input data.
        No model parameters are updated in this stage.
        """

        # Fix random seed to ensure deterministic evaluation
        fix_seed(args.seed)

        # Build loaders (features computed once); evaluate only on the sealed test split,
        # which matches the partition held out during --train (same seed -> identical split).
        _, _, test_loader = build_split_loaders(
            data_file=file_name,
            data_path=data_path,
            transformer_path=args.Transformer_path,
            device=device,
            seed=args.seed,
            max_length=max_length,
            transformer_batch_size=args.batch_size,
        )

        # Initialize model and load saved checkpoint
        model = BRIDGE().to(device)
        model_file = os.path.join(args.model_save_path, file_name + '.pth')

        if not os.path.exists(model_file):
            print('Model file does not exist! Please train first and save the model')
            exit()

        model.load_state_dict(torch.load(model_file))
        model.eval()

        # Define evaluation loss (used only for reporting)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        # Run validation and collect predictions
        met, y_all, p_all = validate(model, device, test_loader, criterion)

        # Extract evaluation metrics
        best_auc = met.auc
        best_acc = met.acc
        best_auprc = met.prc
        best_mcc = met.mcc

        # Print evaluation results
        print(
            "{} auc: {:.4f} acc: {:.4f} auprc: {:.4f} mcc: {:.4f}".format(
                file_name, best_auc, best_acc, best_auprc, best_mcc
            )
        )


    if args.dynamic_predict:
        """
        Run dynamic prediction using a condition model.

        In this mode, a dynamically resolved model checkpoint is loaded and predictions are generated on the test split without retraining.
        """

        # Fix random seed for reproducibility
        fix_seed(args.seed)

        # Resolve the appropriate dynamic model name based on dataset identifier
        model_file = resolve_dynamic_model_name(file_name)
        model_file = os.path.join(args.model_save_path, model_file + '.pth')

        if not os.path.exists(model_file):
            print('Model file does not exitsts! Please train first and save the model')
            exit()

        # Build loaders (features computed once); cross cell-line prediction is evaluated on
        # the sealed test split for consistency with --validate.
        _, _, test_loader = build_split_loaders(
            data_file=file_name,
            data_path=data_path,
            transformer_path=args.Transformer_path,
            device=device,
            seed=args.seed,
            max_length=max_length,
            transformer_batch_size=args.batch_size,
        )

        # Load dynamic BRIDGE model checkpoint
        model = BRIDGE().to(device)
        model.load_state_dict(torch.load(model_file))
        model.eval()

        # Loss is used only for metric computation
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        # Perform prediction and evaluation
        met, y_all, p_all = validate(model, device, test_loader, criterion)

        # Report dynamic prediction performance
        best_auc = met.auc
        best_acc = met.acc
        best_auprc = met.prc
        best_mcc = met.mcc
        print(
            "Dynamic prediction mode. {} auc: {:.4f} acc: {:.4f} "
            "auprc: {:.4f} mcc: {:.4f}".format(
                file_name, best_auc, best_acc, best_auprc, best_mcc
            )
        )


if __name__ == '__main__':
    """
    Command-line interface (CLI) entry point for running BRIDGE.

    This section defines all runtime arguments required to train, validate,
    or perform dynamic prediction with the BRIDGE model. Each argument
    controls a specific aspect of data input, model configuration, or
    execution mode.
    """
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Welcome to BRIDGE!')
    
    # Dataset and path configuration
    parser.add_argument('--data_file', default='AUH_HepG2', type=str, help='RBP to train or validate')
    parser.add_argument('--data_path', default='./dataset', type=str, help='The data path')
    parser.add_argument("--results_dir",default="./results",type=str,help="Root directory for outputs; will create logs/, model/, metrics/ under it")
    parser.add_argument('--Transformer_path', default='./RBPformer', type=str, help='BERT model path, in case you have another BERT')
    parser.add_argument('--model_save_path', default='./results/model', type=str, help='Save the trained model for dynamic prediction')
    parser.add_argument('--batch_size', default=2048, type=int, help='The batch size for BERT embedding generation')
    
    # Execution mode flags
    parser.add_argument('--train', default=False, action='store_true', help='Run training mode')
    parser.add_argument('--validate', default=False, action='store_true', help='Run validation mode')
    parser.add_argument('--dynamic_predict', default=False, action='store_true', help='Run dynamic prediction mode')

    # Output and reproducibility settings
    parser.add_argument('--outdir', default='./results/rsid', type=str, help='Save the output files')
    parser.add_argument('--seed', default=42, type=int, help='The random seed')
    
    # Hardware and optimization settings
    parser.add_argument('--device_num', type=int, default=0, help='The GPU device number to use')
    parser.add_argument('--use_cpu', action='store_true', help='Force using CPU even if GPU is available')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping epochs')
    
    # Parse command-line arguments and launch main pipeline
    args = parser.parse_args()
    main(args)
