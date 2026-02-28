"""
Verify SHA256 checksums of downloaded BRIDGE data and pretrained weights.

Expected input:
- A checksum file (checksums.sha256)
- Local directory containing downloaded files

Usage:
    python verify_checksums.py --checksum checksums.sha256 --data_dir ./
"""

import hashlib
import argparse
from pathlib import Path

def sha256sum(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    with open(args.checksum) as f:
        for line in f:
            expected, fname = line.strip().split()
            file_path = Path(args.data_dir) / fname
            if not file_path.exists():
                raise FileNotFoundError(f"{fname} not found")

            actual = sha256sum(file_path)
            assert actual == expected, f"Checksum mismatch: {fname}"

    print("All checksums verified successfully.")

if __name__ == "__main__":
    main()
