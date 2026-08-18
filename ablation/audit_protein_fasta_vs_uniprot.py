"""
Compare every cached protein FASTA against its resolved UniProt canonical sequence.

Context: docs/BUG_protein_fasta_contamination.md documented 18 RBPs (25/261 datasets) with a
contaminated cached protein FASTA -- wrong protein or a truncated fragment. That audit was
seeded by suspicious header text ("partial") and then verified case by case; it was never an
exhaustive check of all 172 cached files. This script is that exhaustive check: it fetches the
UniProt canonical sequence for every resolved gene (via audit_protein_alphafold_coverage.py's
output) and diffs it against the cached FASTA directly, using the same length-ratio /
k-mer-coverage metric the original audit used.

Two independent numbers matter and must both be checked -- neither alone is sufficient:
  - length_ratio: local_length / uniprot_length. Catches truncation/extension.
  - kmer_coverage: fraction of the LOCAL sequence's 10-mers found anywhere in the UniProt
    sequence. Catches wrong-protein contamination (near 0) vs. a genuinely-correct partial
    sequence (near 1, even if severely truncated by length).
A sequence that is a clean, correct prefix of the real protein scores HIGH on kmer_coverage
(everything present is really there) but LOW on length_ratio (most of it is missing) -- so
classifying on coverage alone mislabels severe truncations as clean matches.

Example:
    python -m ablation.audit_protein_fasta_vs_uniprot \\
        --fasta_dir /quobyte/savirangrp/manav/dataset/protein \\
        --survey_csv docs/protein_alphafold_coverage.csv \\
        --out_csv docs/protein_sequence_comparison.csv
"""
import argparse
import csv
import os
import urllib.error
import urllib.parse
import urllib.request

UNIPROT_BASE = "https://rest.uniprot.org"
KMER_K = 10
BATCH_SIZE = 80

ALIASES = {"eIF4AIII": "EIF4A3", "U2AF65": "U2AF2"}


def read_fasta(path):
    header, seq_lines = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
            else:
                seq_lines.append(line.strip())
    return header, "".join(seq_lines)


def kmer_coverage(query, reference, k=KMER_K):
    """Fraction of `query`'s k-mers found anywhere in `reference`. 1.0 = fully contained."""
    if len(query) < k:
        return float(query in reference)
    q_kmers = {query[i : i + k] for i in range(len(query) - k + 1)}
    if not q_kmers:
        return 0.0
    r_kmers = {reference[i : i + k] for i in range(len(reference) - k + 1)}
    return len(q_kmers & r_kmers) / len(q_kmers)


def fetch_sequences_batch(accessions):
    """Fetch full sequences for many accessions in one stream request each (UniProt's
    /uniprotkb/stream endpoint accepts an OR-query over accessions), instead of one request
    per protein. Returns {accession: seq}."""
    out = {}
    accs = sorted(accessions)
    for i in range(0, len(accs), BATCH_SIZE):
        chunk = accs[i : i + BATCH_SIZE]
        query = "(" + " OR ".join(f"accession:{a}" for a in chunk) + ")"
        q = urllib.parse.urlencode({"query": query, "format": "fasta"})
        url = f"{UNIPROT_BASE}/uniprotkb/stream?{q}"
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        acc, seq = None, []
        for line in text.splitlines():
            if line.startswith(">"):
                if acc and seq:
                    out[acc] = "".join(seq)
                parts = line[1:].split("|")  # >sp|ACCESSION|... or >tr|ACCESSION|...
                acc = parts[1] if len(parts) > 1 else None
                seq = []
            else:
                seq.append(line.strip())
        if acc and seq:
            out[acc] = "".join(seq)
        print(f"  fetched batch {i // BATCH_SIZE + 1}: {len(chunk)} requested, running total {len(out)}")
    return out


def classify(coverage, length_ratio):
    """Both metrics must agree before calling something a clean match -- see module docstring
    for why coverage alone is not sufficient (it does not penalize truncation)."""
    if coverage < 0.5:
        return "WRONG_OR_SEVERE_FRAGMENT"
    if coverage >= 0.99 and 0.97 <= length_ratio <= 1.03:
        return "MATCH"
    return "INCOMPLETE"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta_dir", default="/quobyte/savirangrp/manav/dataset/protein")
    ap.add_argument("--survey_csv", default="docs/protein_alphafold_coverage.csv",
                     help="Output of audit_protein_alphafold_coverage.py; supplies resolved UniProt accessions.")
    ap.add_argument("--out_csv", default="docs/protein_sequence_comparison.csv")
    args = ap.parse_args()

    survey_rows = list(csv.DictReader(open(args.survey_csv)))
    gene_to_acc = {r["gene_symbol"]: r["uniprot_accession"] for r in survey_rows if r["uniprot_accession"]}
    accessions = set(gene_to_acc.values())
    print(f"Fetching {len(accessions)} UniProt sequences in batches...")
    uniprot_seqs = fetch_sequences_batch(accessions)
    print(f"Fetched {len(uniprot_seqs)} sequences.\n")

    local_files = sorted(fn for fn in os.listdir(args.fasta_dir) if fn.endswith(".fasta"))
    print(f"Comparing {len(local_files)} local FASTA files...\n")

    rows = []
    for fn in local_files:
        name = os.path.splitext(fn)[0]
        gene = ALIASES.get(name, name)
        header, local_seq = read_fasta(os.path.join(args.fasta_dir, fn))
        acc = gene_to_acc.get(gene, "")
        row = {
            "local_filename": name, "query_gene": gene, "uniprot_accession": acc,
            "local_header": header or "", "local_length": len(local_seq),
        }
        if not acc:
            row.update({"flag": "NO_UNIPROT_ACCESSION", "uniprot_length": "", "length_ratio": "",
                        "kmer_coverage": "", "exact_substring": ""})
            rows.append(row)
            print(f"{name}: SKIP (no accession)")
            continue
        uni_seq = uniprot_seqs.get(acc, "")
        if not uni_seq:
            row.update({"flag": "UNIPROT_FETCH_FAILED", "uniprot_length": "", "length_ratio": "",
                        "kmer_coverage": "", "exact_substring": ""})
            rows.append(row)
            print(f"{name}: UNIPROT FETCH FAILED for {acc}")
            continue
        cov = kmer_coverage(local_seq, uni_seq)
        ratio = len(local_seq) / len(uni_seq) if uni_seq else 0.0
        exact = local_seq in uni_seq or uni_seq in local_seq
        flag = classify(cov, ratio)
        row.update({
            "uniprot_length": len(uni_seq), "length_ratio": round(ratio, 4),
            "kmer_coverage": round(cov, 4), "exact_substring": exact, "flag": flag,
        })
        rows.append(row)
        marker = "" if flag == "MATCH" else f"  <-- {flag} (cov={cov:.3f}, len {len(local_seq)}/{len(uni_seq)})"
        print(f"{name}: {flag}{marker}")

    fieldnames = ["local_filename", "query_gene", "uniprot_accession", "local_header",
                  "local_length", "uniprot_length", "length_ratio", "kmer_coverage",
                  "exact_substring", "flag"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_match = sum(1 for r in rows if r["flag"] == "MATCH")
    n_incomplete = sum(1 for r in rows if r["flag"] == "INCOMPLETE")
    n_wrong = sum(1 for r in rows if r["flag"] == "WRONG_OR_SEVERE_FRAGMENT")
    n_skip = sum(1 for r in rows if r["flag"] in ("NO_UNIPROT_ACCESSION", "UNIPROT_FETCH_FAILED"))
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")
    print(f"MATCH={n_match} INCOMPLETE={n_incomplete} WRONG_OR_SEVERE_FRAGMENT={n_wrong} SKIPPED={n_skip}")


if __name__ == "__main__":
    main()
