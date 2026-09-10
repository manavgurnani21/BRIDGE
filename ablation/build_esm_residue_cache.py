"""
Offline precompute: per-residue ESM-2 embeddings for the 'attn_protein' ablation config.

Reads FASTA sequences from ``--fasta_dir`` (one file per RBP, e.g. ``AUH.fasta``), runs each
through ESM-2 (``esm2_t33_650M_UR50D``), and writes ``{cache_dir}/{RBP}.npy``: float32, shape
``(P, 1280)``, P = protein length. Keyed by RBP name (not BRIDGE dataset stem) since many
dataset stems share an RBP across cell lines; see
``utils.protein_features.load_protein_residue_embedding`` for the dataset-stem -> RBP lookup
used at train time.

Fixes an off-by-one bug present in PaRPI_BIP's ``utils/esm.py`` (which slices
``token_representations[0, 1:len(seq)-1]``, silently dropping the sequence's last residue):
this script uses ``token_representations[0, 1:len(seq)+1]`` and asserts the result has exactly
``len(seq)`` rows.

Requires the 'fair-esm' package (``pip install fair-esm``), which is NOT part of this repo's
tracked env (``BRIDGE.yml``/``reqs.txt``) since nothing else in the pipeline needs it -- run
this script in a separate env/venv. First run downloads the ~2.5GB ``esm2_t33_650M_UR50D``
checkpoint via torch hub.

Example:
    python -m ablation.build_esm_residue_cache \\
        --fasta_dir /quobyte/savirangrp/manav/dataset/protein \\
        --cache_dir /quobyte/savirangrp/manav/esm_residue
"""

import argparse
import glob
import os

import numpy as np
import torch


def read_fasta(path):
    """Return the single sequence in a FASTA file as a string (header line stripped)."""
    seq_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    return "".join(seq_lines)


def esm_residue_features(model, batch_converter, rbp, seq, device):
    """Return the (P, 1280) per-residue ESM-2 embedding for one protein sequence."""
    _, _, tokens = batch_converter([(rbp, seq)])
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)
    reps = out["representations"][33][0]     # (1 + P + 1, 1280): CLS, residues, EOS
    residue_reps = reps[1 : len(seq) + 1]     # (P, 1280) -- not reps[1:len(seq)-1] (off-by-one)
    assert residue_reps.shape[0] == len(seq), (
        f"{rbp}: expected {len(seq)} residue rows, got {residue_reps.shape[0]}"
    )
    return residue_reps.cpu().numpy().astype(np.float32)


def write_npy_atomic(path, array):
    """Write a .npy file atomically via a temp file + rename (mirrors run_ablation.write_row)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(
        description="Build per-residue ESM-2 embedding cache for the 'attn_protein' config."
    )
    parser.add_argument("--fasta_dir", default="/quobyte/savirangrp/manav/dataset/protein", type=str,
                         help="Dir of {RBP}.fasta files, one per unique RBP.")
    parser.add_argument("--cache_dir", required=True, type=str,
                         help="Output dir for {RBP}.npy per-residue embeddings.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute and overwrite even if {RBP}.npy already exists.")
    parser.add_argument("--device", default="cpu", type=str, help="'cpu' or 'cuda'.")
    args = parser.parse_args()

    import esm  # fair-esm; imported lazily so --help works without the dependency installed

    os.makedirs(args.cache_dir, exist_ok=True)
    fasta_paths = sorted(glob.glob(os.path.join(args.fasta_dir, "*.fasta")))
    if not fasta_paths:
        raise FileNotFoundError(f"No .fasta files found under {args.fasta_dir}")

    device = torch.device(args.device)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    done = skipped = failed = 0
    for fasta_path in fasta_paths:
        rbp = os.path.splitext(os.path.basename(fasta_path))[0]
        out_path = os.path.join(args.cache_dir, f"{rbp}.npy")
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue
        try:
            seq = read_fasta(fasta_path)
            if not seq:
                raise ValueError(f"empty sequence in {fasta_path}")
            residue_reps = esm_residue_features(model, batch_converter, rbp, seq, device)
            write_npy_atomic(out_path, residue_reps)
            done += 1
            print(f"[{rbp}] wrote {residue_reps.shape} -> {out_path}")
        except Exception as e:
            failed += 1
            print(f"[{rbp}] FAILED: {e}")

    print(f"\ndone={done} skipped={skipped} failed={failed} total={len(fasta_paths)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
