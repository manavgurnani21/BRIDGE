"""
Secondary-structure feature utilities: RNAplfold profiles and per-base icSHAPE scores.

This module provides two independent ways to obtain "structure" inputs aligned to nucleotide
positions for downstream models (e.g., BRIDGE):

1) RNAplfold-derived multi-channel profiles (P/E/H/I/M)
   ----------------------------------------------------
   Via :func:`generateStructureFeatures`, this module can run external RNAplfold wrapper
   executables to compute loop-type probabilities per position and parse them into a tensor
   shaped ``(N, L, 5)``.

   Channels (column order in the returned array):
    - ``P``: pairedness probability (computed as residual probability mass)
    - ``H``: hairpin-loop probability
    - ``I``: internal-loop probability
    - ``M``: multi-loop probability
    - ``E``: external-region probability

   The combined profile is cached on disk and parsed by :func:`read_combined_profile`.

2) Per-base structure reactivity strings (icSHAPE / "icshape")
   -----------------------------------------------------------
   Via :func:`build_structure_tensor`, this module can convert per-sequence structure strings
   such as icSHAPE reactivity tracks into a padded numeric tensor shaped ``(N, 1, L)``.

   In this representation, each sequence has a **single** structure channel where the i-th
   value corresponds to the i-th nucleotide position (same length/alignment as the sequence).

   The typical upstream format is a comma-separated string, for example::

       "0.12,0.03,0.50,0.10,..."

   This is referred to as "icshape" in some parts of the codebase.

Who this is for
---------------
- Users preparing token-aligned structure inputs for training/inference.
- Developers maintaining preprocessing and cache behavior.

Input / output conventions
--------------------------
Token alignment
    All structure tensors produced here are position-aligned. The caller is responsible for
    ensuring that the chosen ``L`` matches the sequence length convention used elsewhere
    (e.g., fixed-length 101 in many BRIDGE pipelines).

RNAplfold path (multi-channel)
    - Input: a FASTA file path ``dataset_path`` readable by external wrapper executables.
    - Output: ``np.ndarray`` of shape ``(N, L, 5)`` and ``dtype=float``.

icSHAPE path (single-channel)
    - Input: ``structs`` is a list of comma-separated numeric strings, one per sequence.
    - Output: ``np.ndarray`` of shape ``(N, 1, max_length)`` (float64 by NumPy default).
    - Length constraint: each string must contain exactly ``max_length`` comma-separated values.

External dependency (RNAplfold only)
------------------------------------
:func:`run_RNA` uses ``os.system`` to invoke four wrapper executables under ``script_path``:

- ``E_RNAplfold``, ``H_RNAplfold``, ``I_RNAplfold``, ``M_RNAplfold``

These wrappers are expected to:
- read FASTA from stdin (``< fasta_path``)
- write two-line-per-record profiles to ``*_profile.txt``

.. warning::
   If these executables are missing or not executable, the RNAplfold path will fail.

How to use
----------
RNAplfold multi-channel features:

.. code-block:: python

    feats = generateStructureFeatures(
        dataset_path="inputs.fa",
        script_path="path/to/wrappers",
        basic_path="workdir/struct_cache/",
        W=101, L=101, u=1,
        dataset_name="my_dataset"
    )
    # feats: (N, L, 5) with columns [P, H, I, M, E] after parsing

icSHAPE / icshape single-channel tensor:

.. code-block:: python

    structs = [
        "0.1,0.2,0.3,0.4",
        "0.0,0.5,0.5,0.2",
    ]
    x = build_structure_tensor(structs, max_length=4)
    # x: (2, 1, 4)

Important notes / caveats
-------------------------
- These two structure representations are **not interchangeable**:
  RNAplfold returns 5 loop/pairedness channels, while icSHAPE returns a single per-base score.

- Pairedness computation (RNAplfold):
  ``P = 1 - E - H - I - M`` assumes the four probabilities sum to ``<= 1`` per position.
  If they sum to > 1 (numerical issues or wrapper semantics), P may become negative.

- Cache path check mismatch (behavior preserved):
  ``generateStructureFeatures`` checks for ``basic_path + "/combined_profile.txt"`` but writes
  to ``<basic_path>/<dataset_name>/combined_profile.txt``. Ensure your cache layout matches,
  or adjust the check if you standardize caching.

- Parsing of icSHAPE strings:
  :func:`build_structure_tensor` does not trim whitespace or trailing commas. Upstream strings
  should be clean (e.g., no trailing comma). A mismatch in value count will raise an error
  (or fail assignment/broadcasting).

"""


import argparse  # unused in this module's current functions; kept for CLI-style extension
import os  # directory creation and file-existence checks for the RNAplfold cache
import re  # normalizes whitespace when parsing profile text lines
import linecache  # bulk-reads the combined profile file into memory line by line
import numpy as np  # builds and reshapes the numeric structure tensors
from functools import reduce  # folds a list of values into a tab-delimited string
from collections import OrderedDict  # preserves base-to-vector encoding order (unused directly by build_structure_tensor)
from typing import List  # type hint for build_structure_tensor's input
import numpy as np  # duplicate import, kept as-is (harmless)

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),  # unknown/placeholder base: all-zero one-hot
    ('A', [1, 0, 0, 0]),  # adenine one-hot code
    ('C', [0, 1, 0, 0]),  # cytosine one-hot code
    ('G', [0, 0, 1, 0]),  # guanine one-hot code
    ('T', [0, 0, 0, 1]),  # thymine one-hot code
    ('N', [0.25, 0.25, 0.25, 0.25]),  # ambiguous base: uniform probability over the 4 bases
])

seq_encoding_keys = list(encoding_seq.keys())  # ordered list of base symbols, e.g. for index lookups
seq_encoding_vectors = np.array(list(encoding_seq.values()))  # matching one-hot/uniform vectors as a NumPy array


def mk_dir(dir):
    """
    Create a directory.
    """
    try:
        os.makedirs(dir)  # create the directory (and any missing parents)
    except OSError:
        print('Can not make directory:', dir)  # directory likely already exists; log and continue rather than crash


def list_to_str(lst):
    '''
    Convert a list of values into a tab-delimited string.
    '''
    return reduce((lambda s, f: s + '\t' + str(f)), lst, '')  # fold every value onto the accumulator string, tab-separated


def concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region):
    """
    Combine multiple whitespace-delimited structure tracks into a per-position feature matrix.

    Args:
        pairedness (str):
            Whitespace-delimited numeric tokens for the pairedness (P) track.
        hairpin_loop (str):
            Whitespace-delimited numeric tokens for the hairpin-loop (H) track.
        internal_loop (str):
            Whitespace-delimited numeric tokens for the internal-loop (I) track.
        multi_loop (str):
            Whitespace-delimited numeric tokens for the multi-loop (M) track.
        external_region (str):
            Whitespace-delimited numeric tokens for the external-region (E) track.

    Returns:
        np.ndarray:
            Array of shape (L, 5) where L is the number of positions/tokens and columns
            correspond to [P, H, I, M, E] in the order provided to this function.

    Notes:
        - Each input string is split with `.split()`; multiple spaces are treated as separators.
        - All input tracks are assumed to have the same token length L.
    """
    combine_list = [pairedness.split(), hairpin_loop.split(), internal_loop.split(), multi_loop.split(),
                    external_region.split()]  # tokenize each whitespace-delimited track into a list of strings
    return np.array(combine_list).T  # stack tracks as columns and transpose to (L, 5) position-major layout


def defineExperimentPaths(basic_path, name_id):
    """
    Create and return directory paths used for RNAplfold-derived structure profiles.

    This function creates the following directories under:
        basic_path/<name_id>/
            E/, H/, I/, M/

    Args:
        basic_path (str):
            Root directory for storing intermediate outputs.
        name_id (str or int):
            Dataset identifier appended to basic_path.

    Returns:
        Tuple[str, str, str, str, str]:
            (path, E_path, H_path, I_path, M_path), each ending with '/'.
    """
    path = basic_path + str(name_id) + '/'  # root directory for this dataset's cached profiles
    E_path = basic_path + str(name_id) + '/E/'  # external-region profile output directory
    H_path = basic_path + str(name_id) + '/H/'  # hairpin-loop profile output directory
    I_path = basic_path + str(name_id) + '/I/'  # internal-loop profile output directory
    M_path = basic_path + str(name_id) + '/M/'  # multi-loop profile output directory
    mk_dir(E_path)  # ensure the E subdirectory exists
    mk_dir(H_path)  # ensure the H subdirectory exists
    mk_dir(I_path)  # ensure the I subdirectory exists
    mk_dir(M_path)  # ensure the M subdirectory exists
    return path, E_path, H_path, I_path, M_path  # all five paths, for run_RNA and generateStructureFeatures to write into


def read_combined_profile(file_path):
    """
    Parse a combined structure profile file into a numeric tensor.

    Expected file format:
        The file is assumed to contain repeating 6-line blocks:
            line 0: an identifier line (ignored by this parser)
            line 1: pairedness probabilities (P) as whitespace-separated numbers
            line 2: hairpin-loop probabilities (H)
            line 3: internal-loop probabilities (I)
            line 4: multi-loop probabilities (M)
            line 5: external-region probabilities (E)

        This function reads lines 1..5 of each block and concatenates them into an (L, 5) array.

    Args:
        file_path (str):
            Path to the combined profile text file.

    Returns:
        np.ndarray:
            Float array of shape (N, L, 5), where:
                N = number of records (blocks),
                L = number of positions/tokens in the profile lines,
                5 = number of structure channels.

    Notes:
        - Uses `linecache.getlines`, which reads the whole file into memory.
        - Whitespace is normalized with `re.sub('[\\s+]', ' ', ...)` before splitting.
        - Assumes every record is exactly 6 lines and all profile lines have equal token length.
    """
    i = 0  # index of the current 6-line block's identifier line
    secondary_structure_list = []  # accumulates one (L, 5) array per record
    filelines = linecache.getlines(file_path)  # read the whole combined-profile file into memory as a list of lines
    file_length = len(filelines)  # total number of lines, used to bound the block loop
    while i <= file_length - 1:  # iterate one 6-line record block at a time
        pairedness = re.sub('[\s+]', ' ', filelines[i + 1].strip())  # normalize whitespace in the P track line
        hairpin_loop = re.sub('[\s+]', ' ', filelines[i + 2].strip())  # normalize whitespace in the H track line
        internal_loop = re.sub('[\s+]', ' ', filelines[i + 3].strip())  # normalize whitespace in the I track line
        multi_loop = re.sub('[\s+]', ' ', filelines[i + 4].strip())  # normalize whitespace in the M track line
        external_region = re.sub('[\s+]', ' ', filelines[i + 5].strip())  # normalize whitespace in the E track line
        combine_array = concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region)  # merge the 5 tracks into one (L, 5) array
        secondary_structure_list.append(combine_array)  # keep this record's array
        i = i + 6  # advance to the next 6-line block

    return np.array(secondary_structure_list).astype(float)  # stack all records into (N, L, 5) and cast to float


# def definecombinePaths(basic_path, name_id):
#     path = basic_path + str(name_id) + '/'
#     E_path = basic_path + str(name_id) + '/E/'
#     H_path = basic_path + str(name_id) + '/H/'
#     I_path = basic_path + str(name_id) + '/I/'
#     M_path = basic_path + str(name_id) + '/M/'
#     return path, E_path, H_path, I_path, M_path


def run_RNA(fasta_path, script_path, E_path, H_path, I_path, M_path, W, L, u):
    """
    Run external RNAplfold wrapper executables to generate structure profile text files.

    This function invokes four commands via `os.system`:
        - <script_path>/E_RNAplfold ...
        - <script_path>/H_RNAplfold ...
        - <script_path>/I_RNAplfold ...
        - <script_path>/M_RNAplfold ...

    Each command reads from stdin redirected from `fasta_path` and writes output to:
        E_path/E_profile.txt, H_path/H_profile.txt, I_path/I_profile.txt, M_path/M_profile.txt

    Args:
        fasta_path (str):
            Path to an input FASTA file for RNAplfold to process.
        script_path (str):
            Directory containing the RNAplfold wrapper executables.
        E_path (str):
            Output directory for E_profile.txt.
        H_path (str):
            Output directory for H_profile.txt.
        I_path (str):
            Output directory for I_profile.txt.
        M_path (str):
            Output directory for M_profile.txt.
        W (int):
            RNAplfold window size argument (-W).
        L (int):
            RNAplfold maximum base pair span argument (-L).
        u (int):
            RNAplfold "unpaired" length argument (-u).

    Returns:
        None.
    """
    os.system(
        script_path + '/E_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        E_path + 'E_profile.txt')  # shell out to the external-region RNAplfold wrapper, writing E_profile.txt
    os.system(
        script_path + '/H_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        H_path + 'H_profile.txt')  # shell out to the hairpin-loop RNAplfold wrapper, writing H_profile.txt
    os.system(
        script_path + '/I_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        I_path + 'I_profile.txt')  # shell out to the internal-loop RNAplfold wrapper, writing I_profile.txt
    os.system(
        script_path + '/M_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        M_path + 'M_profile.txt')  # shell out to the multi-loop RNAplfold wrapper, writing M_profile.txt


def generateStructureFeatures(dataset_path, script_path, basic_path, W, L, u, dataset_name=''):
    """
    Generate per-position RNA secondary-structure features using RNAplfold and cache results.

    Workflow:
        1) Create output directories under: basic_path/<dataset_name>/[E,H,I,M]/
        2) If `<basic_path>/combined_profile.txt` does NOT exist, run RNAplfold wrappers and
           write a combined profile file at: <path>/combined_profile.txt
        3) Parse the combined profile file into a numeric tensor via `read_combined_profile`.

    Args:
        dataset_path (str):
            Path to input FASTA file to process.
        script_path (str):
            Directory containing RNAplfold wrapper executables.
        basic_path (str):
            Root directory for intermediate outputs and cache files.
        W (int):
            RNAplfold window size (-W).
        L (int):
            RNAplfold maximum base pair span (-L).
        u (int):
            RNAplfold unpaired length (-u).
        dataset_name (str, optional):
            Identifier used to create subdirectories under basic_path. Default: ''.

    Returns:
        np.ndarray:
            Structure feature tensor of shape (N, L, 5), dtype float,
            as returned by `read_combined_profile`.

    Notes:
        - The cache existence check currently uses `basic_path + '/combined_profile.txt'` while
          the file is written to `path + 'combined_profile.txt'` (where path=basic_path/<dataset_name>/).
          This behavior is preserved; ensure your basic_path/dataset_name usage matches expectation.
        - Pairedness is computed as:
              P_prob = 1 - E - H - I - M
          assuming the four probabilities sum to <= 1 per position.
    """
    path, E_path, H_path, I_path, M_path = defineExperimentPaths(
        basic_path, dataset_name)  # create/locate the per-channel output directories for this dataset
    if not os.path.exists(basic_path+'/combined_profile.txt'):  # only regenerate if the (mismatched-path) cache marker is absent
        run_RNA(dataset_path, script_path, E_path, H_path, I_path, M_path, W=W, L=L, u=u)  # invoke RNAplfold wrappers to produce the 4 raw profile files
        fEprofile = open(E_path + 'E_profile.txt')  # open the external-region profile output
        Eprofiles = fEprofile.readlines()  # read all lines (id + probability line pairs) for E

        fHprofile = open(H_path + 'H_profile.txt')  # open the hairpin-loop profile output
        Hprofiles = fHprofile.readlines()  # read all lines for H

        fIprofile = open(I_path + 'I_profile.txt')  # open the internal-loop profile output
        Iprofiles = fIprofile.readlines()  # read all lines for I

        fMprofile = open(M_path + 'M_profile.txt')  # open the multi-loop profile output
        Mprofiles = fMprofile.readlines()  # read all lines for M

        mw = int(1)  # minimum window offset; used to trim the first (mw-1) positions from each track

        fhout = open(path + 'combined_profile.txt', 'w')  # output file that will hold the merged 6-line-per-record profile

        for i in range(0, int(len(Eprofiles) / 2)):  # each record occupies 2 lines (id, probabilities) in every profile file
            id = Eprofiles[i * 2].split()[0]  # this record's identifier, taken from the E profile's id line
            print(id, file=fhout)  # write the identifier line for this record
            E_prob = Eprofiles[i * 2 + 1].split()  # this record's external-region probabilities, one token per position
            H_prob = Hprofiles[i * 2 + 1].split()  # this record's hairpin-loop probabilities
            I_prob = Iprofiles[i * 2 + 1].split()  # this record's internal-loop probabilities
            M_prob = Mprofiles[i * 2 + 1].split()  # this record's multi-loop probabilities
            P_prob = list(
                map((lambda a, b, c, d: 1 - float(a) - float(b) - float(c) - float(d)), E_prob, H_prob, I_prob, M_prob))  # derive pairedness as the residual probability mass not accounted for by E/H/I/M
            print(list_to_str(P_prob[mw - 1:len(P_prob)]), file=fhout)  # write the (possibly trimmed) pairedness track
            print(list_to_str(E_prob[mw - 1:len(P_prob)]), file=fhout)  # write the (possibly trimmed) external-region track
            print(list_to_str(H_prob[mw - 1:len(P_prob)]), file=fhout)  # write the (possibly trimmed) hairpin-loop track
            print(list_to_str(I_prob[mw - 1:len(P_prob)]), file=fhout)  # write the (possibly trimmed) internal-loop track
            print(list_to_str(M_prob[mw - 1:len(P_prob)]), file=fhout)  # write the (possibly trimmed) multi-loop track
        fhout.close()  # flush and close the combined profile file

    features = read_combined_profile(path + 'combined_profile.txt')  # parse the (now-guaranteed-to-exist) combined profile into a tensor
    return features  # (N, L, 5) structure feature tensor for this dataset


def build_structure_tensor(structs: List[str], max_length: int) -> np.ndarray:
    """
    Convert comma-separated structure score strings into a padded 3D tensor.

    Args:
        structs (List[str]):
            List of comma-separated numeric strings, one per sequence.
            Example: "0.1,0.2,0.3,..."
        max_length (int):
            Expected sequence length. Each `structs[i]` should contain exactly `max_length`
            comma-separated values.

    Returns:
        np.ndarray:
            Array of shape (N, 1, max_length), dtype float64 (due to np.zeros default),
            containing the parsed structure values.

    Raises:
        ValueError:
            If a structure string cannot be parsed into floats.
        ValueError or broadcasting error:
            If the number of values is not equal to `max_length` (assignment will fail).
    """
    structure = np.zeros((len(structs), 1, max_length))  # preallocate the output tensor: one row per sequence, single structure channel
    for i in range(len(structs)):  # process each sequence's structure string independently
        struct = structs[i].split(',')  # split the comma-separated reactivity/score string into tokens
        ti = [float(t) for t in struct]  # parse each token to a float structure value
        ti = np.array(ti).reshape(1, -1)  # reshape to (1, max_length) to match the single-channel layout
        structure[i] = np.concatenate([ti], axis=0)  # write this sequence's channel into the preallocated tensor
    return structure  # (N, 1, max_length) padded/parsed structure tensor
