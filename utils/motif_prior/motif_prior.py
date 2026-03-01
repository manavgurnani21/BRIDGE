"""
Motif-prior generation and loading utilities (STREME-based pipeline wrapper).

This module wraps an external motif discovery/prior tool (invoked via
`utils/motif_prior/get_motif_prior`) and exposes two main utilities:

  1) `get_motif_prior(data_file)`:
       Ensures the motif prior artifacts exist on disk for a given dataset ID.
       It performs a lightweight existence/emptiness check and only runs the
       external pipeline when needed.

  2) `get_motif_prior_matrix(data_file) -> np.ndarray`:
       Calls `get_motif_prior`, then loads the generated tab file and converts
       it into a model-ready NumPy tensor with an explicit channel dimension.

Typical usage context
---------------------
Use this module in training/inference pipelines where a model expects an
additional "motif prior" feature channel derived from positive/negative
sequence sets. This is common when combining:
  - sequence-based encoders (CNN/Transformer),
  - structure/shape channels (e.g., icSHAPE), and
  - motif-derived priors from motif discovery tools.

Core functions
--------------
1) `extract_sequences_from_fa(input_fa: str) -> List[str]`
   Input:
     - FASTA-like file where each record spans 3 lines:
         line 0: header (e.g., >id ...)
         line 1: sequence
         line 2: icSHAPE scores (or any auxiliary line)
   Output:
     - list of sequence strings (line 1 of each 3-line record)
   Notes:
     - This reader assumes fixed 3-line blocks per record; it will silently
       ignore trailing incomplete blocks.

2) `get_motif_prior(data_file: str) -> None`
   Input:
     - `data_file`: dataset identifier, e.g., "LIN28B_HEK293"
   Disk inputs (expected):
     - dataset/{data_file}_pos.fa
     - dataset/{data_file}_neg.fa
   Disk outputs (checked/produced):
     - utils/motif_prior/output/{data_file}/output/STRME_training_set.tab
   Behavior:
     - If the target directory/file does not exist, or the file is empty,
       run the external binary to generate motif priors.
     - Otherwise, skip computation.
   Side effects:
     - Creates an output directory under `utils/motif_prior/out/{data_file}`
     - Creates temporary files containing foreground/background sequences
     - Runs a subprocess (raises on failure with `check=True`)
     - Deletes the temporary files after completion

   External dependency:
     - Executable: `utils/motif_prior/get_motif_prior` must be present and runnable.

3) `get_motif_prior_matrix(data_file: str) -> np.ndarray`
   Input:
     - `data_file`: dataset identifier (same as above)
   Output:
     - NumPy array of shape (N, 1, M)
       where:
         N = number of sequences/examples in the training set file
         1 = channel dimension (for downstream CNN-style pipelines)
         M = number of motif features per example (columns excluding the ID column)
   File parsed:
     - utils/motif_prior/output/{data_file}/output/STRME_training_set.tab
       tab-delimited with a header row; first column is an ID, remaining columns
       are numeric motif features.

Typical usage
-------------
# 1) Ensure motif priors exist on disk (cached if already present)
get_motif_prior("LIN28B_HEK293")

# 2) Load motif prior tensor for model input
motif = get_motif_prior_matrix("LIN28B_HEK293")
# motif.shape == (N, 1, M)
"""

import os
import subprocess
import numpy as np
import tempfile
from typing import List

def extract_sequences_from_fa(input_fa: str) -> List[str]:
    """
    Extract pure sequence lines from a fasta file containing header, sequence, and icSHAPE scores.

    Args:
        input_fa (str): Path to the input FASTA file.

    Returns:
        List[str]: A list of sequence strings extracted from the FASTA file.
    """
    with open(input_fa, 'r') as fin:
        lines = fin.readlines()
        sequences = []
        for i in range(0, len(lines), 3):
            if i + 1 < len(lines):
                seq = lines[i + 1].strip()
                sequences.append(seq)
        return sequences


def get_motif_prior(data_file: str) -> None:
    """
    Run motif prior if the output file is missing or empty.

    Parameters
    ----------
    data_file : str
        Identifier for the dataset, e.g., "LIN28B_HEK293".
    """
    base_dir = "utils/motif_prior/output"
    target_folder = os.path.join(base_dir, data_file)
    target_file = os.path.join(target_folder, "output", "STRME_training_set.tab")

    should_run = False

    if not os.path.isdir(target_folder):
        print(f"Directory does not exist: {target_folder}")
        should_run = True
    else:
        if not os.path.isfile(target_file):
            print(f"Motif file does not exist: {target_file}")
            should_run = True
        elif os.path.getsize(target_file) == 0:
            print(f"Motif file is empty: {target_file}")
            should_run = True
        else:
            print(f"Valid motif file detected, skipping motif_prior: {target_file}")

    if not should_run:
        return

    output_directory = os.path.join("utils/motif_prior/out", data_file)
    os.makedirs(output_directory, exist_ok=True)

    fg_seqs = extract_sequences_from_fa(f"dataset/{data_file}_pos.fa")
    bg_seqs = extract_sequences_from_fa(f"dataset/{data_file}_neg.fa")

    # Create temp files to store them
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as fg_temp, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as bg_temp:

        fg_temp.write("\n".join(fg_seqs) + "\n")
        bg_temp.write("\n".join(bg_seqs) + "\n")
        fg_temp_path = fg_temp.name
        bg_temp_path = bg_temp.name

    # Build and run motif_prior command
    command = [
        "utils/motif_prior/get_motif_prior",
        "-fg", fg_temp_path,
        "-bg", bg_temp_path,
        "-o", output_directory,
        "-stremeP", "100",
        "-logregP", "0",
        "-testP", "0",
        "-alph", "1,2,3,5,6,7"
    ]

    print(f"Executing motif_prior for {data_file}...")
    subprocess.run(command, check=True)
    print(f"motif_prior completed for {data_file}")

    # Optional: cleanup temp files
    os.remove(fg_temp_path)
    os.remove(bg_temp_path)


def get_motif_prior_matrix(data_file: str) -> np.ndarray:
    """
    Generate and load the motif prior matrix.

    This function calls `get_motif_prior` to create a motif prior file 
    for the given dataset, reads the generated file, extracts motif 
    feature values (excluding the first ID column), and returns them 
    as a 3D NumPy array with an added channel dimension.

    Args:
        data_file (str): Name of the dataset file (without extension) 
                         used to generate motif priors.

    Returns:
        np.ndarray: A 3D NumPy array of shape (N, 1, M), where N is the number of samples, 1 is the channel dimension, and M is the motif feature length.
    """
    get_motif_prior(data_file)
    trainfile = f'utils/motif_prior/output/{data_file}/output/STRME_training_set.tab'
    trainset = np.loadtxt(trainfile, delimiter='\t', skiprows=1)
    X = trainset[:, 1:]
    motif = np.zeros((len(X), X.shape[1]))
    for i in range(len(X)):
        motif[i, :len(X[i])] = X[i, :]
    motif = np.expand_dims(motif, axis=1)
    return motif
