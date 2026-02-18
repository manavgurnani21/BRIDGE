import argparse
import os
import re
import linecache
import numpy as np
from functools import reduce
from collections import OrderedDict
from typing import List
import numpy as np

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))


def mk_dir(dir):
    """
    Create a directory.
    """
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def list_to_str(lst):
    '''
    Convert a list of values into a tab-delimited string.
    '''
    return reduce((lambda s, f: s + '\t' + str(f)), lst, '')


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
                    external_region.split()]
    return np.array(combine_list).T


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
    path = basic_path + str(name_id) + '/'
    E_path = basic_path + str(name_id) + '/E/'
    H_path = basic_path + str(name_id) + '/H/'
    I_path = basic_path + str(name_id) + '/I/'
    M_path = basic_path + str(name_id) + '/M/'
    mk_dir(E_path)
    mk_dir(H_path)
    mk_dir(I_path)
    mk_dir(M_path)
    return path, E_path, H_path, I_path, M_path


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
    i = 0
    secondary_structure_list = []
    filelines = linecache.getlines(file_path)
    file_length = len(filelines)
    while i <= file_length - 1:
        pairedness = re.sub('[\s+]', ' ', filelines[i + 1].strip())
        hairpin_loop = re.sub('[\s+]', ' ', filelines[i + 2].strip())
        internal_loop = re.sub('[\s+]', ' ', filelines[i + 3].strip())
        multi_loop = re.sub('[\s+]', ' ', filelines[i + 4].strip())
        external_region = re.sub('[\s+]', ' ', filelines[i + 5].strip())
        combine_array = concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region)
        secondary_structure_list.append(combine_array)
        i = i + 6

    return np.array(secondary_structure_list).astype(float)


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
        E_path + 'E_profile.txt')
    os.system(
        script_path + '/H_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        H_path + 'H_profile.txt')
    os.system(
        script_path + '/I_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        I_path + 'I_profile.txt')
    os.system(
        script_path + '/M_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        M_path + 'M_profile.txt')


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
        basic_path, dataset_name)
    if not os.path.exists(basic_path+'/combined_profile.txt'):
        run_RNA(dataset_path, script_path, E_path, H_path, I_path, M_path, W=W, L=L, u=u)
        fEprofile = open(E_path + 'E_profile.txt')
        Eprofiles = fEprofile.readlines()

        fHprofile = open(H_path + 'H_profile.txt')
        Hprofiles = fHprofile.readlines()

        fIprofile = open(I_path + 'I_profile.txt')
        Iprofiles = fIprofile.readlines()

        fMprofile = open(M_path + 'M_profile.txt')
        Mprofiles = fMprofile.readlines()

        mw = int(1)

        fhout = open(path + 'combined_profile.txt', 'w')

        for i in range(0, int(len(Eprofiles) / 2)):
            id = Eprofiles[i * 2].split()[0]
            print(id, file=fhout)
            E_prob = Eprofiles[i * 2 + 1].split()
            H_prob = Hprofiles[i * 2 + 1].split()
            I_prob = Iprofiles[i * 2 + 1].split()
            M_prob = Mprofiles[i * 2 + 1].split()
            P_prob = list(
                map((lambda a, b, c, d: 1 - float(a) - float(b) - float(c) - float(d)), E_prob, H_prob, I_prob, M_prob))
            print(list_to_str(P_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(E_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(H_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(I_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(M_prob[mw - 1:len(P_prob)]), file=fhout)
        fhout.close()

    features = read_combined_profile(path + 'combined_profile.txt')
    return features


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
    structure = np.zeros((len(structs), 1, max_length))
    for i in range(len(structs)):
        struct = structs[i].split(',')
        ti = [float(t) for t in struct]
        ti = np.array(ti).reshape(1, -1)
        structure[i] = np.concatenate([ti], axis=0)
    return structure
