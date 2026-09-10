import os  # OS path/environment utilities (file existence checks, env vars, path joins)
import random  # Python's stdlib PRNG, seeded for reproducibility
import argparse  # CLI argument parsing
import subprocess  # imported but unused here; kept for parity with original script
from pathlib import Path  # object-oriented filesystem path handling used for run directories
import time  # wall-clock timing of the training run
from datetime import datetime  # timestamping run names
import json  # writing config/metrics summaries to disk

import numpy as np  # numerical arrays; also seeded for reproducibility
import torch  # PyTorch core (tensors, autograd, CUDA control)
import torch.nn as nn  # neural network building blocks (loss functions, layers)

from utils.BRIDGE import BRIDGE  # the BRIDGE model class (RNA+protein multimodal architecture)
from utils.train_loop import validate, fit_bridge  # shared training/eval loop: fit_bridge trains+early-stops, validate scores a loader
from utils.utils import param_num, resolve_dynamic_model_name  # param_num prints model size; resolve_dynamic_model_name maps a dataset id to a cross-condition checkpoint name
from utils.data_pipeline import build_split_loaders  # builds train/val/test DataLoaders from raw dataset files (embeddings + features computed once)


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
        from termcolor import cprint  # optional dependency providing colored console printing
    except ImportError:
        cprint = None  # termcolor not installed; fall back to plain print below

    # Attempt to import pycrayon (optional; not required for basic printing)
    try:
        from pycrayon import CrayonClient  # optional dependency, not actually used beyond the import
    except ImportError:
        CrayonClient = None  # pycrayon not installed; harmless since it's unused

    # Use colored printing if available; otherwise fall back to plain print
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)  # print with requested color/background/attributes
    else:
        print(text)  # no color support available; print plain text


def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)  # no seed given: pick a random one so the run is still reproducible if logged
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)  # seed Python's random module (used for e.g. shuffling)
    os.environ['PYTHONHASHSEED'] = str(seed)  # fix hash seed so hash-based iteration order is reproducible across runs
    np.random.seed(seed)  # seed NumPy's legacy global RNG
    torch.manual_seed(seed)  # seed PyTorch's CPU RNG
    torch.cuda.manual_seed(seed)  # seed PyTorch's CUDA RNG for the current device
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
    results_dir = Path(getattr(args, "results_dir", "./results"))  # root output directory (default ./results, overridable via --results_dir)
    logs_dir = results_dir/"logs"  # subdirectory for per-run log files
    metrics_dir = results_dir/"metrics"  # subdirectory for best-metric JSON summaries

    # If user passed --model_save_path, prefer it; otherwise use results_dir/model
    model_dir = Path(getattr(args, "model_save_path", "")) if getattr(args, "model_save_path", "") else (results_dir / "model")  # checkpoint output directory

    logs_dir.mkdir(parents=True, exist_ok=True)  # create logs dir (and parents) if missing
    metrics_dir.mkdir(parents=True, exist_ok=True)  # create metrics dir (and parents) if missing
    model_dir.mkdir(parents=True, exist_ok=True)  # create model checkpoint dir (and parents) if missing

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")  # timestamp used to make each run's outputs unique
    run_name = f"{file_name}_{run_id}"  # unique run identifier combining dataset name and timestamp
    return run_name, logs_dir, model_dir, metrics_dir  # hand back the resolved paths/name for use in main()


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
        device = torch.device("cpu")  # force CPU execution regardless of GPU availability
    else:
        device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")  # use requested GPU if available, else fall back to CPU

    # Explicitly set the CUDA device if GPU is used
    if device.type == 'cuda':
        torch.cuda.set_device(args.device_num)  # make this the active CUDA device for subsequent tensor allocations

    # Maximum sequence length used for padding/truncation
    max_length = 101

    # Dataset identifier and base data directory
    file_name = args.data_file  # which RBP/dataset to train or evaluate on
    data_path = args.data_path  # root directory containing the dataset files
    Transformer_batch_size = args.batch_size  # batch size used when running the pretrained transformer to extract embeddings

    if args.train:
        # Start timing the full training procedure
        start_time = time.time()  # record wall-clock start time for reporting total training duration

        # prepare run dirs + open logfile
        run_name, logs_dir, model_dir, metrics_dir = _prepare_run_dirs(args, file_name)  # create/resolve output directories for this run
        log_path = logs_dir / f"{run_name}.log"  # path to this run's plain-text log file
        log_fp = open(log_path, "a", encoding="utf-8")  # open the log file for appending

        def log_both(msg: str, color=None, attrs=None):
            # write to file
            log_fp.write(msg + "\n")  # persist the message to the run's log file
            log_fp.flush()  # ensure the message is written to disk immediately (useful if the run crashes)
            # print to console
            log_print(msg, color=color, attrs=attrs)  # also print the message to stdout with optional coloring

        # Write config file (args + key hyperparams)
        config = {
            "run_name": run_name,  # unique identifier for this run
            "data_file": args.data_file,  # dataset/RBP identifier used
            "data_path": args.data_path,  # root data directory used
            "Transformer_path": args.Transformer_path,  # path to the pretrained transformer used for embeddings
            "seed": args.seed,  # random seed used for this run
            "use_cpu": bool(args.use_cpu),  # whether CPU-only execution was forced
            "device": str(device),  # resolved torch device string (e.g. "cuda:0" or "cpu")
            "device_num": int(args.device_num),  # requested GPU index
            "train": bool(args.train),  # whether training mode was requested
            "validate": bool(getattr(args, "validate", False)),  # whether validate mode was also requested
            "dynamic_predict": bool(getattr(args, "dynamic_predict", False)),  # whether dynamic-predict mode was also requested
            "max_length": int(max_length),  # sequence length used for padding/truncation
            "lr_cli": float(args.lr),  # learning rate passed on the command line
            "early_stopping": int(args.early_stopping)  # early-stopping patience in epochs
        }


        # Build train/val/test loaders (features computed once via the shared pipeline).
        # Early stopping + checkpoint selection watch the validation set; the sealed test
        # split is left untouched here and reserved for --validate / --dynamic_predict.
        train_loader, val_loader, _ = build_split_loaders(
            data_file=file_name,  # dataset identifier to load
            data_path=data_path,  # root directory containing the dataset files
            transformer_path=args.Transformer_path,  # pretrained transformer used to embed RNA sequences
            device=device,  # device on which embeddings/features are computed
            seed=args.seed,  # seed controlling the train/val/test split
            max_length=max_length,  # sequence length used for padding/truncation
            transformer_batch_size=Transformer_batch_size,  # batch size for transformer embedding extraction
        )
        # Initialize the BRIDGE model
        model = BRIDGE().to(device)  # construct the BRIDGE architecture and move its parameters to the target device

        # Binary classification loss with class imbalance compensation
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))  # binary cross-entropy with logits; positive class weighted 2x to offset class imbalance

        # Adam optimizer with weight decay regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)  # Adam optimizer over all model parameters

        # Learning rate scheduling parameters
        initial_lrate = 0.0016  # LR used once warmup completes, before step decay begins
        drop = 0.8  # multiplicative factor applied to LR at each decay step
        epochs_drop = 5.0  # number of epochs between successive LR decay steps
        warmup_epochs = 40  # number of initial epochs during which LR is linearly warmed up

        # include schedule/loss/optimizer info in config
        config.update(
            {
                "lr_schedule": {
                    "initial_lrate": float(initial_lrate),  # record post-warmup base LR
                    "drop": float(drop),  # record decay factor
                    "epochs_drop": float(epochs_drop),  # record decay interval
                    "warmup_epochs": int(warmup_epochs),  # record warmup duration
                }
            }
        )

        config_path = logs_dir / f"{run_name}_config.json"  # path where this run's config snapshot is saved
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)  # persist the full run configuration as pretty-printed JSON

        log_both(f"[RUN] {run_name}", color="green", attrs=["bold"])  # announce the run name in the log/console
        log_both(f"[DIR] logs={logs_dir} model={model_dir} metrics={metrics_dir}")  # record where outputs are being written
        log_both(f"[CFG] {config_path}")  # record where the config snapshot was saved

        # Print total number of model parameters
        param_num(model)  # log the parameter count of the BRIDGE model for reference

        # Train with the shared loop: warm-up + step-decay LR, val-AUC checkpoint selection,
        # and early stopping. The best checkpoint is saved to {model_dir}/{run_name}.pth.
        best = fit_bridge(
            model, device, train_loader, val_loader, criterion, optimizer,
            max_epochs=200,  # upper bound on training epochs (early stopping usually halts sooner)
            warmup_epochs=warmup_epochs,  # epochs of LR warmup before decay schedule kicks in
            initial_lrate=initial_lrate,  # base LR after warmup
            drop=drop,  # step-decay multiplicative factor
            epochs_drop=epochs_drop,  # step-decay interval in epochs
            early_stopping=args.early_stopping,  # patience (epochs without val-AUC improvement) before stopping
            ckpt_path=model_dir / f"{run_name}.pth",  # where to save the best checkpoint found during training
            log_fn=lambda m: log_both(m, color="green", attrs=["bold"]),  # callback used by fit_bridge to emit per-epoch logs
            tag=file_name,  # dataset tag used in log messages
        )
        best_auc = best["best_val_auc"]  # best validation ROC-AUC achieved during training
        best_acc = best["best_val_acc"]  # validation accuracy at the best checkpoint
        best_mcc = best["best_val_mcc"]  # validation Matthews correlation coefficient at the best checkpoint
        best_prc = best["best_val_prc"]  # validation PR-AUC at the best checkpoint
        best_epoch = best["best_epoch"]  # epoch index at which the best checkpoint was recorded

        # Report best validation performance
        # print("{} auc: {:.4f} acc: {:.4f} prc: {:.4f} mcc: {:.4f}".format(file_name, best_auc, best_acc, best_prc, best_mcc))
        summary_line = (
            f"{file_name} best: auc={best_auc:.4f} acc={best_acc:.4f} prc={best_prc:.4f} mcc={best_mcc:.4f} "  # human-readable summary of best validation metrics
            f"(epoch={best_epoch})"  # epoch at which these metrics were achieved
        )
        log_both(summary_line, color="green", attrs=["bold"])  # write the summary line to log file and console

        best_summary = {
            "run_name": run_name,  # identifier for this training run
            "data_file": file_name,  # dataset/RBP this run trained on
            "best_epoch": int(best_epoch),  # epoch of the best checkpoint
            "best_val_auc": float(best_auc),  # best validation AUC
            "best_val_acc": float(best_acc),  # validation accuracy at best checkpoint
            "best_val_prc": float(best_prc),  # validation PR-AUC at best checkpoint
            "best_val_mcc": float(best_mcc),  # validation MCC at best checkpoint
            "seed": int(args.seed),  # seed used for this run
            "device": str(device),  # device the run executed on
            "checkpoint": str((model_dir / f"{run_name}.pth").resolve()),  # absolute path to the saved model checkpoint
            "log_file": str(log_path.resolve()),  # absolute path to this run's log file
            "config_file": str(config_path.resolve()),  # absolute path to this run's config JSON
        }
        best_path = metrics_dir / f"{run_name}_best.json"  # path where the best-metrics summary is saved
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)  # persist the best-metrics summary as pretty-printed JSON
        log_both(f"[BEST] {best_path}")  # record where the best-metrics summary was saved

        log_fp.close()  # close the run's log file handle

        # Report total training time
        end_time = time.time()  # wall-clock end time
        time_cost = end_time - start_time  # total elapsed training time in seconds
        print("Time cost: {:.2f} min".format(time_cost / 60))  # print total training duration in minutes


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
            data_file=file_name,  # dataset identifier to load
            data_path=data_path,  # root directory containing the dataset files
            transformer_path=args.Transformer_path,  # pretrained transformer used to embed RNA sequences
            device=device,  # device on which embeddings/features are computed
            seed=args.seed,  # seed controlling the split (must match the training run's seed)
            max_length=max_length,  # sequence length used for padding/truncation
            transformer_batch_size=args.batch_size,  # batch size for transformer embedding extraction
        )

        # Initialize model and load saved checkpoint
        model = BRIDGE().to(device)  # construct a fresh BRIDGE model instance on the target device
        model_file = os.path.join(args.model_save_path, file_name + '.pth')  # expected path to the trained checkpoint for this dataset

        if not os.path.exists(model_file):
            print('Model file does not exist! Please train first and save the model')  # warn the user that no checkpoint was found
            exit()  # abort since there is nothing to evaluate

        model.load_state_dict(torch.load(model_file))  # load the trained weights into the model
        model.eval()  # switch to evaluation mode (disables dropout, freezes batch-norm stats)

        # Define evaluation loss (used only for reporting)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))  # same loss as training, used here only to compute a reportable loss value

        # Run validation and collect predictions
        met, y_all, p_all = validate(model, device, test_loader, criterion)  # run the model over the test loader and compute metrics/predictions

        # Extract evaluation metrics
        best_auc = met.auc  # ROC-AUC on the test split
        best_acc = met.acc  # accuracy on the test split
        best_auprc = met.prc  # PR-AUC on the test split
        best_mcc = met.mcc  # Matthews correlation coefficient on the test split

        # Print evaluation results
        print(
            "{} auc: {:.4f} acc: {:.4f} auprc: {:.4f} mcc: {:.4f}".format(
                file_name, best_auc, best_acc, best_auprc, best_mcc  # format test-set metrics for console output
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
        model_file = resolve_dynamic_model_name(file_name)  # map this dataset/cell-line to the checkpoint trained on a related condition
        model_file = os.path.join(args.model_save_path, model_file + '.pth')  # full path to that cross-condition checkpoint

        if not os.path.exists(model_file):
            print('Model file does not exitsts! Please train first and save the model')  # warn that the resolved checkpoint is missing
            exit()  # abort since there is nothing to predict with

        # Build loaders (features computed once); cross cell-line prediction is evaluated on
        # the sealed test split for consistency with --validate.
        _, _, test_loader = build_split_loaders(
            data_file=file_name,  # dataset identifier to load
            data_path=data_path,  # root directory containing the dataset files
            transformer_path=args.Transformer_path,  # pretrained transformer used to embed RNA sequences
            device=device,  # device on which embeddings/features are computed
            seed=args.seed,  # seed controlling the split
            max_length=max_length,  # sequence length used for padding/truncation
            transformer_batch_size=args.batch_size,  # batch size for transformer embedding extraction
        )

        # Load dynamic BRIDGE model checkpoint
        model = BRIDGE().to(device)  # construct a fresh BRIDGE model instance on the target device
        model.load_state_dict(torch.load(model_file))  # load the cross-condition checkpoint's weights
        model.eval()  # switch to evaluation mode

        # Loss is used only for metric computation
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))  # same loss as training, used here only for reportable metrics

        # Perform prediction and evaluation
        met, y_all, p_all = validate(model, device, test_loader, criterion)  # run the cross-condition model over the test loader

        # Report dynamic prediction performance
        best_auc = met.auc  # ROC-AUC of the cross-condition prediction
        best_acc = met.acc  # accuracy of the cross-condition prediction
        best_auprc = met.prc  # PR-AUC of the cross-condition prediction
        best_mcc = met.mcc  # MCC of the cross-condition prediction
        print(
            "Dynamic prediction mode. {} auc: {:.4f} acc: {:.4f} "
            "auprc: {:.4f} mcc: {:.4f}".format(
                file_name, best_auc, best_acc, best_auprc, best_mcc  # format cross-condition metrics for console output
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
    parser = argparse.ArgumentParser(description='Welcome to BRIDGE!')  # sets up the CLI parser with a friendly description

    # Dataset and path configuration
    parser.add_argument('--data_file', default='AUH_HepG2', type=str, help='RBP to train or validate')  # which dataset/RBP to use
    parser.add_argument('--data_path', default='./dataset', type=str, help='The data path')  # root directory of input data
    parser.add_argument("--results_dir",default="./results",type=str,help="Root directory for outputs; will create logs/, model/, metrics/ under it")  # where all run outputs are written
    parser.add_argument('--Transformer_path', default='./RBPformer', type=str, help='BERT model path, in case you have another BERT')  # path to the pretrained RNA transformer used for embeddings
    parser.add_argument('--model_save_path', default='./results/model', type=str, help='Save the trained model for dynamic prediction')  # directory to save/load model checkpoints
    parser.add_argument('--batch_size', default=2048, type=int, help='The batch size for BERT embedding generation')  # batch size used specifically for transformer embedding extraction

    # Execution mode flags
    parser.add_argument('--train', default=False, action='store_true', help='Run training mode')  # enable the training branch of main()
    parser.add_argument('--validate', default=False, action='store_true', help='Run validation mode')  # enable the validate-on-test-split branch of main()
    parser.add_argument('--dynamic_predict', default=False, action='store_true', help='Run dynamic prediction mode')  # enable the cross-condition prediction branch of main()

    # Output and reproducibility settings
    parser.add_argument('--outdir', default='./results/rsid', type=str, help='Save the output files')  # (unused directly in main(), reserved for downstream output scripts)
    parser.add_argument('--seed', default=42, type=int, help='The random seed')  # random seed for reproducibility

    # Hardware and optimization settings
    parser.add_argument('--device_num', type=int, default=0, help='The GPU device number to use')  # which CUDA device index to use
    parser.add_argument('--use_cpu', action='store_true', help='Force using CPU even if GPU is available')  # force CPU execution
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')  # initial learning rate passed to the optimizer
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping epochs')  # patience for early stopping

    # Parse command-line arguments and launch main pipeline
    args = parser.parse_args()  # parse sys.argv into a Namespace of the options above
    main(args)  # run the selected pipeline stage(s) with the parsed configuration
