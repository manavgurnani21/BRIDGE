"""
Resolve authoritative protein FASTA sequences for K562/HepG2 RBPs via ENCODE + UniProt.

Context: the protein FASTAs inherited from PaRPI_BIP (``--fasta_dir``, one file per RBP) have
no documented source -- neither the PaRPI paper, its GitHub repo, nor its Zenodo release
records where the sequences came from. A subset is known-defective (wrong protein / wrong
organism / truncated isoform; see ``docs/BUG_protein_fasta_contamination.md``), and since there
is no original source to fall back to, this script builds a documented, verifiable one instead
of trying to recover PaRPI's undocumented process.

Method, per RBP:
  1. Query the ENCODE portal for an eCLIP experiment on that RBP in K562 or HepG2 (BRIDGE's own
     dataset stems tell us which cell line(s) to expect a hit in). ENCODE's ``target`` ->
     ``genes[]`` record carries a hard-asserted ``organism`` plus ``dbxrefs`` (UniProtKB /
     RefSeq / Ensembl gene ID) -- this is what resolves naming ambiguity (e.g. which AGO
     paralog, which species) that a plain name search cannot.
  2. Query UniProt for the reviewed (Swiss-Prot) canonical human entry for that gene symbol.
     Cross-check the returned accession against ENCODE's own dbxrefs list when present, for a
     second, independent confirmation.
  3. Compare the resolved sequence to the currently-cached FASTA (length ratio + k-mer
     coverage, same metric used in the original contamination audit) and write everything to a
     report -- nothing under ``--fasta_dir`` is modified. Resolved sequences are written to
     ``--out_dir`` for manual review before anyone promotes them into the real cache.

Only covers K562/HepG2 by design (ENCODE eCLIP was run almost exclusively in those two cell
lines; HEK293/Hela datasets in this collection are not ENCODE experiments -- see the two zero
hits documented in the accompanying discussion, ``AGO``/``MOV10`` under HEK293).

Example:
    python -m ablation.resolve_protein_fasta \\
        --manifest ablation/datasets.txt \\
        --fasta_dir /quobyte/savirangrp/manav/dataset/protein \\
        --out_dir /quobyte/savirangrp/manav/BRIDGE_protein_fasta_resolved
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ENCODE_BASE = "https://www.encodeproject.org"
UNIPROT_BASE = "https://rest.uniprot.org"
CELL_LINES = ("K562", "HepG2")
KMER_K = 10
REQUEST_TIMEOUT = 15
REQUEST_DELAY_S = 0.34  # ~3 req/s, polite to both APIs


def _get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url):
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read().decode("utf-8")


def read_fasta(path):
    """Return (header, sequence) for the single record in a FASTA file."""
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


def rbps_from_manifest(manifest_path):
    """Unique RBP names for K562/HepG2 dataset stems in a BRIDGE ``datasets.txt`` manifest."""
    rbp_cell_lines = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            stem = line.strip()
            if not stem:
                continue
            parts = stem.split("_")
            cell_line = parts[-1]
            if cell_line not in CELL_LINES:
                continue
            rbp = "_".join(parts[:-1])
            rbp_cell_lines.setdefault(rbp, set()).add(cell_line)
    return rbp_cell_lines


def find_encode_target(rbp, cell_lines):
    """Return (matched_cell_line, target_at_id) for the first ENCODE eCLIP hit, or (None, None)."""
    for cell_line in cell_lines:
        query = urllib.parse.urlencode(
            {
                "type": "Experiment",
                "assay_title": "eCLIP",
                "target.label": rbp,
                "biosample_ontology.term_name": cell_line,
                "format": "json",
            }
        )
        try:
            result = _get_json(f"{ENCODE_BASE}/search/?{query}")
        except urllib.error.URLError:
            continue
        graph = result.get("@graph", [])
        if graph:
            target = graph[0].get("target", {})
            target_id = target.get("@id") if isinstance(target, dict) else target
            if target_id:
                return cell_line, target_id
    return None, None


def encode_gene_dbxrefs(target_id):
    """Return (organism, set-of-dbxrefs) from an ENCODE target's linked Gene record(s)."""
    try:
        target = _get_json(f"{ENCODE_BASE}{target_id}?format=json")
    except urllib.error.URLError:
        return None, set()
    dbxrefs = set()
    organism = None
    for gene in target.get("genes", []):
        organism = (gene.get("organism") or "").rstrip("/").rsplit("/", 1)[-1] or organism
        dbxrefs.update(gene.get("dbxrefs", []))
    return organism, dbxrefs


def uniprot_reviewed_canonical(gene_symbol, retries=3):
    """Return (accession, sequence, header, n_hits) for the reviewed human Swiss-Prot entry.

    Queries structured JSON (not raw FASTA text) so hits can be filtered to ones whose
    *primary* gene symbol matches the query exactly -- UniProt's `gene:` search also matches
    synonyms (e.g. querying "NIP7" also returns CCT7, whose gene record lists a "NIP7-1"
    synonym), so relevance-ranking alone is not a safe way to pick the right entry.
    Retries on empty/malformed responses -- observed transient truncated bodies under
    sustained sequential querying that don't raise a URLError but leave nothing parseable.
    """
    query = urllib.parse.urlencode(
        {
            "query": f"gene:{gene_symbol} AND organism_id:9606 AND reviewed:true",
            "format": "json",
            "fields": "accession,gene_primary,sequence",
            "size": "10",
        }
    )
    url = f"{UNIPROT_BASE}/uniprotkb/search?{query}"
    for attempt in range(retries):
        try:
            data = _get_json(url)
        except (urllib.error.URLError, json.JSONDecodeError):
            time.sleep(REQUEST_DELAY_S)
            continue
        results = data.get("results", [])
        if results:
            break
        time.sleep(REQUEST_DELAY_S)
    else:
        return None

    n_hits = len(results)
    exact = [
        r for r in results
        if r.get("genes", [{}])[0].get("geneName", {}).get("value", "").upper() == gene_symbol.upper()
    ]
    chosen = exact[0] if exact else results[0]
    accession = chosen["primaryAccession"]
    seq = chosen["sequence"]["value"]
    if not seq:
        return None
    header = f"{accession} ({chosen.get('genes', [{}])[0].get('geneName', {}).get('value', gene_symbol)})"
    # Only flag as ambiguous if the exact-symbol filter didn't uniquely resolve it.
    ambiguous_count = n_hits if not exact else len(exact)
    return accession, seq, header, ambiguous_count


def resolve_one(rbp, cell_lines, fasta_dir):
    row = {
        "rbp": rbp,
        "cell_lines": ",".join(sorted(cell_lines)),
        "encode_cell_line_hit": "",
        "encode_target": "",
        "encode_organism": "",
        "uniprot_accession": "",
        "uniprot_n_reviewed_hits": "",
        "dbxref_cross_check": "",
        "old_length": "",
        "new_length": "",
        "kmer_coverage_old_in_new": "",
        "length_ratio_old_over_new": "",
        "flag": "",
    }

    old_path = os.path.join(fasta_dir, f"{rbp}.fasta")
    old_seq = ""
    if os.path.exists(old_path):
        _, old_seq = read_fasta(old_path)
        row["old_length"] = len(old_seq)
    else:
        row["flag"] = "NO_EXISTING_FASTA"

    cell_line_hit, target_id = find_encode_target(rbp, sorted(cell_lines))
    time.sleep(REQUEST_DELAY_S)
    if not target_id:
        row["flag"] = (row["flag"] + ";NO_ENCODE_HIT").lstrip(";")
        return row, None
    row["encode_cell_line_hit"] = cell_line_hit
    row["encode_target"] = target_id

    organism, dbxrefs = encode_gene_dbxrefs(target_id)
    time.sleep(REQUEST_DELAY_S)
    row["encode_organism"] = organism or ""
    if organism and organism != "human":
        row["flag"] = (row["flag"] + ";ENCODE_NON_HUMAN_TARGET").lstrip(";")

    uni = uniprot_reviewed_canonical(rbp)
    time.sleep(REQUEST_DELAY_S)
    if uni is None:
        row["flag"] = (row["flag"] + ";NO_UNIPROT_REVIEWED_HIT").lstrip(";")
        return row, None
    accession, new_seq, _header, n_hits = uni
    row["uniprot_accession"] = accession
    row["uniprot_n_reviewed_hits"] = n_hits
    if n_hits > 1:
        row["flag"] = (row["flag"] + ";MULTIPLE_REVIEWED_HITS_TOOK_FIRST").lstrip(";")

    row["dbxref_cross_check"] = "MATCH" if any(accession in d for d in dbxrefs) else "NO_ENCODE_DBXREF_MATCH"
    if row["dbxref_cross_check"] == "NO_ENCODE_DBXREF_MATCH" and dbxrefs:
        row["flag"] = (row["flag"] + ";DBXREF_MISMATCH").lstrip(";")

    row["new_length"] = len(new_seq)
    if old_seq:
        # Two independent questions: is old's content genuinely found in new (identity), and
        # does old cover new's full length (completeness). A clean truncation scores ~1.0 on
        # the first and low on the second -- conflating them mislabels e.g. a 42%-length but
        # otherwise-correct fragment as "resolved" instead of "truncated".
        cov = kmer_coverage(old_seq, new_seq)
        length_ratio = len(old_seq) / len(new_seq) if new_seq else 0.0
        row["kmer_coverage_old_in_new"] = round(cov, 4)
        row["length_ratio_old_over_new"] = round(length_ratio, 4)
        if cov < 0.5:
            row["flag"] = (row["flag"] + ";LIKELY_WRONG_PROTEIN").lstrip(";")
        elif length_ratio < 0.95:
            row["flag"] = (row["flag"] + ";OLD_WAS_TRUNCATED").lstrip(";")
    if not row["flag"]:
        exact = old_seq and row.get("kmer_coverage_old_in_new", 0) >= 0.999 and row.get("length_ratio_old_over_new", 0) >= 0.999
        row["flag"] = "OK_NO_CHANGE_DETECTED" if exact else "RESOLVED"

    return row, (accession, new_seq)


def main():
    parser = argparse.ArgumentParser(
        description="Resolve K562/HepG2 protein FASTAs via ENCODE target metadata + UniProt "
        "canonical sequence; writes a report and staged sequences, never overwrites --fasta_dir."
    )
    parser.add_argument("--manifest", default="ablation/datasets.txt", type=str)
    parser.add_argument("--fasta_dir", default="/quobyte/savirangrp/manav/dataset/protein", type=str)
    parser.add_argument("--out_dir", required=True, type=str,
                         help="Where to write resolved_protein_fasta_report.csv and {RBP}.fasta staging files.")
    parser.add_argument("--rbp", type=str, default=None,
                         help="Resolve a single RBP instead of the full K562/HepG2 manifest (for spot checks).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rbp_cell_lines = rbps_from_manifest(args.manifest)
    if args.rbp:
        rbp_cell_lines = {args.rbp: rbp_cell_lines.get(args.rbp, set(CELL_LINES))}

    report_path = os.path.join(args.out_dir, "resolved_protein_fasta_report.csv")
    fieldnames = [
        "rbp", "cell_lines", "encode_cell_line_hit", "encode_target", "encode_organism",
        "uniprot_accession", "uniprot_n_reviewed_hits", "dbxref_cross_check",
        "old_length", "new_length", "kmer_coverage_old_in_new", "length_ratio_old_over_new", "flag",
    ]
    import csv

    with open(report_path, "w", newline="", encoding="utf-8") as report_f:
        writer = csv.DictWriter(report_f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (rbp, cell_lines) in enumerate(sorted(rbp_cell_lines.items()), 1):
            print(f"[{i}/{len(rbp_cell_lines)}] {rbp} ({','.join(sorted(cell_lines))})")
            row, resolved = resolve_one(rbp, cell_lines, args.fasta_dir)
            writer.writerow(row)
            report_f.flush()
            print(f"    -> {row['flag']}")
            if resolved:
                accession, new_seq = resolved
                staged_path = os.path.join(args.out_dir, f"{rbp}.fasta")
                with open(staged_path, "w", encoding="utf-8") as fh:
                    fh.write(f">{accession} {rbp} (UniProt reviewed canonical, resolved via ENCODE+UniProt)\n")
                    for j in range(0, len(new_seq), 70):
                        fh.write(new_seq[j : j + 70] + "\n")

    print(f"\nDone. Report: {report_path}")
    print(f"Staged sequences: {args.out_dir}/*.fasta (NOT applied to {args.fasta_dir})")


if __name__ == "__main__":
    main()
