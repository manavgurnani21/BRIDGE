#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Script: count_shared_binding_sites.sh
#
# Purpose
# -------
# For each RBP, find shared binding sites between every pair of cell lines
# (using BED interval overlap), then SUM shared counts across all RBPs per pair.
#
# Output
# ------
# A CSV file named: intersection_results.csv
# with columns:
#   Interaction,Count
# where:
#   Interaction is "CellA&CellB" (no spaces)
#   Count       is the summed number of intersected intervals across all RBPs
#
# Expected Input Files
# --------------------
# BED files located under DATA_DIR, named as:
#   <RBP>_<CELL_LINE>.bed
#
# Examples:
#   DDX3X_K562.bed
#   DDX3X_HepG2.bed
#   NCBP2_HEK293T.bed
#
# Requirements
# ------------
# - bedtools in PATH
#
# Notes about intersection
# ------------------------
# bedtools intersect:
#   -a FILE1 -b FILE2
# outputs each interval from A that overlaps any interval in B.
# If an interval in A overlaps multiple intervals in B, it can be output multiple
# times (depending on B), increasing counts.
#
# If you want a "unique A intervals that overlap B at least once", use:
#   bedtools intersect -u -a FILE1 -b FILE2 | wc -l
###############################################################################


#######################################
# User-configurable parameters
#######################################

# Cell lines (order matters only for pair naming; we only do i<j combinations)
CELL_LINES=("K562" "HepG2" "H9" "HEK293" "HEK293T" "Hela")

# Directory containing BED files
DATA_DIR="/home/wangyubo/code/preprocess_data/RBP_binding_sites/cross_cell_type_bedfiles"

# Output CSV
OUT_CSV="intersection_results.csv"

# Choose intersection mode:
#   "raw"    : bedtools intersect output lines count (default; matches your original)
#   "unique" : count unique A intervals that overlap B at least once (bedtools -u)
INTERSECT_MODE="raw"


#######################################
# Helper: print to stderr
#######################################
log() {
  # Usage: log "message"
  echo "[INFO] $*" >&2
}


#######################################
# Helper: check dependencies
#######################################
check_deps() {
  if ! command -v bedtools >/dev/null 2>&1; then
    echo "[ERROR] bedtools not found in PATH. Please load/install bedtools." >&2
    exit 1
  fi

  # bash version check (associative arrays require bash 4+)
  if (( BASH_VERSINFO[0] < 4 )); then
    echo "[ERROR] This script requires bash 4+ (associative arrays)." >&2
    exit 1
  fi

  if [[ ! -d "$DATA_DIR" ]]; then
    echo "[ERROR] DATA_DIR not found: $DATA_DIR" >&2
    exit 1
  fi
}


#######################################
# Helper: extract unique RBP names from BED filenames
#
# Logic:
#   - list *.bed
#   - strip suffix: _<anything>.bed  -> keep <RBP>
#   - unique
#
# Output:
#   prints rbp names, one per line
#######################################
get_rbps() {
  ls "$DATA_DIR" 2>/dev/null \
    | grep -E '\.bed$' \
    | sed -E 's/_.*\.bed$//' \
    | sort -u
}


#######################################
# Helper: create all unique cell-line pairs (i<j)
# Output: "CellA_CellB" keys, one per line
#######################################
get_pairs_keys() {
  local i j
  for ((i=0; i<${#CELL_LINES[@]}; i++)); do
    for ((j=i+1; j<${#CELL_LINES[@]}; j++)); do
      echo "${CELL_LINES[i]}_${CELL_LINES[j]}"
    done
  done
}


#######################################
# Helper: compute intersection count for one file pair
#
# Parameters:
#   $1: file1 (BED)
#   $2: file2 (BED)
#   $3: mode ("raw" or "unique")
#
# Output:
#   prints integer count to stdout
#######################################
intersection_count() {
  local f1="$1"
  local f2="$2"
  local mode="$3"

  if [[ "$mode" == "unique" ]]; then
    bedtools intersect -u -a "$f1" -b "$f2" | wc -l
  else
    # raw mode
    bedtools intersect -a "$f1" -b "$f2" | wc -l
  fi
}


#######################################
# Main workflow
#######################################
main() {
  check_deps

  log "DATA_DIR = $DATA_DIR"
  log "OUT_CSV  = $OUT_CSV"
  log "MODE     = $INTERSECT_MODE"

  # Read RBPs into an array (safe with whitespace-free names)
  mapfile -t RBPS < <(get_rbps)

  if (( ${#RBPS[@]} == 0 )); then
    echo "[ERROR] No .bed files found in: $DATA_DIR" >&2
    exit 1
  fi

  log "Found ${#RBPS[@]} RBPs."

  # Initialize associative array counts[pair_key]=0
  declare -A COUNTS
  while read -r pair_key; do
    COUNTS["$pair_key"]=0
  done < <(get_pairs_keys)

  # Loop over each RBP and each pair of cell lines
  local rbp i j file1 file2 pair_key shared_count
  for rbp in "${RBPS[@]}"; do
    # For each RBP, check all i<j combinations
    for ((i=0; i<${#CELL_LINES[@]}; i++)); do
      for ((j=i+1; j<${#CELL_LINES[@]}; j++)); do
        file1="${DATA_DIR}/${rbp}_${CELL_LINES[i]}.bed"
        file2="${DATA_DIR}/${rbp}_${CELL_LINES[j]}.bed"

        # Only compute if both exist
        if [[ -f "$file1" && -f "$file2" ]]; then
          shared_count="$(intersection_count "$file1" "$file2" "$INTERSECT_MODE")"

          # Optional per-RBP debug prints (comment out if too verbose)
          log "RBP=${rbp}  ${CELL_LINES[i]}&${CELL_LINES[j]}  shared=${shared_count}"

          pair_key="${CELL_LINES[i]}_${CELL_LINES[j]}"
          COUNTS["$pair_key"]=$(( COUNTS["$pair_key"] + shared_count ))
        fi
      done
    done
  done

  # Write CSV
  # Format:
  # Interaction,Count
  # K562&HepG2,123
  log "Writing CSV to $OUT_CSV"
  {
    echo "Interaction,Count"
    for ((i=0; i<${#CELL_LINES[@]}; i++)); do
      for ((j=i+1; j<${#CELL_LINES[@]}; j++)); do
        pair_key="${CELL_LINES[i]}_${CELL_LINES[j]}"
        interaction="${CELL_LINES[i]}&${CELL_LINES[j]}"
        echo "${interaction},${COUNTS["$pair_key"]}"
      done
    done
  } > "$OUT_CSV"

  log "Done. Preview:"
  head -n 10 "$OUT_CSV" >&2
}

main "$@"
