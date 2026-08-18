"""
Survey: for every RBP protein FASTA file BRIDGE actually reads, resolve a reviewed human
UniProt canonical entry, then check the AlphaFold EBI API for a structure prediction under
that accession.

Context: docs/BUG_protein_fasta_contamination.md established that a subset of the cached
protein FASTAs are wrong or truncated. Before any structure-derived feature (AlphaFold contact
maps, pLDDT/disorder, DSSP) can be built on top of the protein branch, we need to know how many
of the 172 cached proteins even have an AlphaFold structure available once resolved to a correct
UniProt sequence. This script answers that -- it does not touch the cached FASTAs themselves.

Companion script: audit_protein_fasta_vs_uniprot.py does the complementary check (does the
*cached* sequence actually match the UniProt sequence this script resolves).

Example:
    python -m ablation.audit_protein_alphafold_coverage \\
        --fasta_dir /quobyte/savirangrp/manav/dataset/protein \\
        --out_csv docs/protein_alphafold_coverage.csv
"""
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

UNIPROT_BASE = "https://rest.uniprot.org"
ALPHAFOLD_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
REQUEST_TIMEOUT = 15
REQUEST_DELAY_S = 0.34

# Filenames in --fasta_dir that are legacy/alternate names, not valid HGNC symbols for
# UniProt's gene: search -> map to the canonical symbol for querying purposes only. The
# original filename is still reported as its own "protein name" in the coverage list.
ALIASES = {
    "eIF4AIII": "EIF4A3",
    "U2AF65": "U2AF2",
}


def _get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def protein_names_from_fasta_dir(fasta_dir):
    return sorted(os.path.splitext(fn)[0] for fn in os.listdir(fasta_dir) if fn.endswith(".fasta"))


def normalize(protein_names):
    query_to_sources = {}
    for name in protein_names:
        gene = ALIASES.get(name, name)
        query_to_sources.setdefault(gene, set()).add(name)
    return query_to_sources


def uniprot_reviewed_canonical(gene_symbol, retries=3):
    query = urllib.parse.urlencode(
        {
            "query": f"gene:{gene_symbol} AND organism_id:9606 AND reviewed:true",
            "format": "json",
            "fields": "accession,gene_primary,sequence,protein_name",
            "size": "10",
        }
    )
    url = f"{UNIPROT_BASE}/uniprotkb/search?{query}"
    for _ in range(retries):
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
    chosen = exact[0] if exact else (results[0] if results else None)
    if chosen is None:
        return None
    accession = chosen["primaryAccession"]
    seq = chosen["sequence"]["value"]
    pname = chosen.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
    ambiguous_count = n_hits if not exact else len(exact)
    return {
        "accession": accession,
        "length": len(seq),
        "protein_name": pname,
        "n_hits": n_hits,
        "ambiguous": ambiguous_count > 1,
    }


def alphafold_prediction(accession):
    """Picks the entry with the largest residue span -- AlphaFold DB splits proteins over
    ~2700aa into multiple overlapping fragment entries, and the API's own ordering is not
    guaranteed to put the most-complete fragment first."""
    url = f"{ALPHAFOLD_API_BASE}/{accession}"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"found": False, "http_status": 404}
        return {"found": False, "http_status": e.code, "error": str(e)}
    except urllib.error.URLError as e:
        return {"found": False, "error": str(e)}
    if not data:
        return {"found": False, "http_status": 200, "note": "empty list"}
    entry = max(data, key=lambda e: (e.get("uniprotEnd") or 0) - (e.get("uniprotStart") or 0))
    return {
        "found": True,
        "model_created": entry.get("modelCreatedDate"),
        "uniprot_start": entry.get("uniprotStart"),
        "uniprot_end": entry.get("uniprotEnd"),
        "latest_version": entry.get("latestVersion"),
        "n_fragments": len(data),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta_dir", default="/quobyte/savirangrp/manav/dataset/protein")
    ap.add_argument("--out_csv", default="docs/protein_alphafold_coverage.csv")
    args = ap.parse_args()

    protein_names = protein_names_from_fasta_dir(args.fasta_dir)
    print(f"Protein FASTA files in {args.fasta_dir}: {len(protein_names)}")

    query_to_sources = normalize(protein_names)
    print(f"Unique UniProt query gene symbols after alias collapsing: {len(query_to_sources)}")

    rows = []
    for i, gene in enumerate(sorted(query_to_sources)):
        sources = sorted(query_to_sources[gene])
        row = {"gene_symbol": gene, "source_rbp_names": ";".join(sources)}
        uni = uniprot_reviewed_canonical(gene)
        time.sleep(REQUEST_DELAY_S)
        if uni is None:
            row.update({"uniprot_accession": "", "uniprot_flag": "NO_REVIEWED_HIT"})
            rows.append(row)
            print(f"[{i+1}/{len(query_to_sources)}] {gene}: NO UNIPROT HIT")
            continue
        row.update({
            "uniprot_accession": uni["accession"],
            "uniprot_length": uni["length"],
            "uniprot_protein_name": uni["protein_name"],
            "uniprot_n_hits": uni["n_hits"],
            "uniprot_ambiguous": uni["ambiguous"],
            "uniprot_flag": "",
        })

        af = alphafold_prediction(uni["accession"])
        time.sleep(REQUEST_DELAY_S)
        row["alphafold_found"] = af.get("found")
        row["alphafold_model_created"] = af.get("model_created", "")
        row["alphafold_uniprot_start"] = af.get("uniprot_start", "")
        row["alphafold_uniprot_end"] = af.get("uniprot_end", "")
        row["alphafold_latest_version"] = af.get("latest_version", "")
        row["alphafold_n_fragments"] = af.get("n_fragments", "")
        row["alphafold_http_status"] = af.get("http_status", "")
        rows.append(row)
        print(f"[{i+1}/{len(query_to_sources)}] {gene} -> {uni['accession']} | AlphaFold found={af.get('found')}")

    fieldnames = [
        "gene_symbol", "source_rbp_names", "uniprot_accession", "uniprot_length",
        "uniprot_protein_name", "uniprot_n_hits", "uniprot_ambiguous", "uniprot_flag",
        "alphafold_found", "alphafold_model_created", "alphafold_uniprot_start",
        "alphafold_uniprot_end", "alphafold_latest_version", "alphafold_n_fragments",
        "alphafold_http_status",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
