import numpy as np
import collections

def get_1_trids():
    """
    Build an index mapping for all possible 1-mers over RNA alphabet {A, C, G, U}.

    Returns
    -------
    dict[str, int]
        Mapping from 1-mer string to integer index (size = 4).
        Example: {'A': 0, 'C': 1, 'G': 2, 'U': 3} (exact ordering depends on generation logic).
    """
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 1

    # Enumerate all length-1 combinations in a deterministic order
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        nucle_com.append(ch0)

    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_2_trids():
    """
    Build an index mapping for all possible 2-mers over RNA alphabet {A, C, G, U}.

    Returns
    -------
    dict[str, int]
        Mapping from 2-mer string to integer index (size = 4^2 = 16).
    """
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 2

    # Enumerate all length-2 combinations in a deterministic order
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        nucle_com.append(ch0 + ch1)

    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_3_trids():
    """
    Build an index mapping for all possible 3-mers over RNA alphabet {A, C, G, U}.

    Returns
    -------
    dict[str, int]
        Mapping from 3-mer string to integer index (size = 4^3 = 64).
    """
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 3

    # Enumerate all length-3 combinations in a deterministic order
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)

    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def frequency(seq, kmer, coden_dict):
    """
    Count occurrences of each k-mer along a sequence and return as an index→count dictionary.

    This helper converts DNA 'T' to RNA 'U' before looking up the k-mer index.
    The returned dictionary is sparse (only k-mers observed in the sequence appear).

    Parameters
    ----------
    seq : str
        Input nucleotide sequence. 'T' will be internally treated as 'U'.
    kmer : int
        k-mer size.
    coden_dict : dict[str, int]
        Mapping from k-mer string to integer index (created by get_1_trids/get_2_trids/get_3_trids).

    Returns
    -------
    dict[int, int]
        Dictionary mapping k-mer index to its occurrence count in the sequence.
    """
    Value = []
    k = kmer

    # Slide a window of size k across the sequence (stride = 1)
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i + k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)

    # Count occurrences of each k-mer index
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def coden(seq, kmer, tris):
    """
    Construct a position-wise k-mer feature tensor of shape (101, |V|).

    For each position i (0-based), this tensor places the *global frequency* of the
    k-mer starting at i into the corresponding k-mer index column. This yields a
    sparse matrix where each row has at most one non-zero entry.

    IMPORTANT DESIGN NOTE
    ---------------------
    This implementation encodes *global counts* at each position (not a one-hot).
    The counts are normalized by 100 (consistent with the original feature scaling
    in the provided implementation).

    Parameters
    ----------
    seq : str
        Input nucleotide sequence (will treat 'T' as 'U').
    kmer : int
        k-mer size (1, 2, or 3).
    tris : dict[str, int]
        k-mer -> index mapping.

    Returns
    -------
    np.ndarray
        Array of shape (101, len(tris)) containing position-wise frequency features.
        Rows beyond the valid k-mer start positions remain all zeros.
    """
    coden_dict = tris

    # Pre-compute global k-mer counts across the entire sequence
    freq_dict = frequency(seq, kmer, coden_dict)

    # Fixed-length representation: always allocate 101 rows
    vectors = np.zeros((101, len(coden_dict.keys())))

    # Slide k-mer window across the sequence and fill position-wise features
    for i in range(len(seq) - int(kmer) + 1):
        # Convert DNA 'T' to RNA 'U' to match the k-mer vocabulary
        value = freq_dict[coden_dict[seq[i:i+kmer].replace('T', 'U')]]

        # Place the normalized global count at the corresponding position and k-mer column
        vectors[i][coden_dict[seq[i:i+kmer].replace('T', 'U')]] = value/100

    return vectors


def processFastaFile(seq):
    """
    Encode a nucleotide sequence into a 3-channel representation with padding up to length 101.

    Encoding scheme (phys_dic)
    --------------------------
    A -> [1, 1, 1]
    U -> [0, 0, 1]
    C -> [0, 1, 0]
    G -> [1, 0, 0]

    Padding rule
    ------------
    For padded positions (i >= seqLength), the last channel is set to 1.
    This acts as an explicit padding indicator so the model can distinguish
    real residues from padded positions.

    Parameters
    ----------
    seq : str
        Input sequence consisting of {A, U, C, G} (after upstream normalization).

    Returns
    -------
    np.ndarray
        Array of shape (101, 3).
    """
    phys_dic = {
        'A': [1, 1, 1],
        'U': [0, 0, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0]
    }

    seqLength = len(seq)
    sequence_vector = np.zeros([101, 3])

    # Encode observed positions
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i]]

    # Mark padded positions explicitly
    for i in range(seqLength, 101):
        sequence_vector[i, -1] = 1

    return sequence_vector


def dpcp(seq):
    """
    Encode dinucleotide physicochemical properties (DPCP) into an 11-dimensional feature vector.

    For each position i, this function assigns the property vector of the dinucleotide seq[i:i+2].
    Positions near the end that do not have a full dinucleotide remain zeros.

    Parameters
    ----------
    seq : str
        Input RNA sequence (A/U/C/G).

    Returns
    -------
    np.ndarray
        Array of shape (101, 11). Only positions 0..len(seq)-2 are filled.
    """
    phys_dic = {
        #Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
        'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
        'AU': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
        'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24,  0.14,],
        'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14,  -7.6, -19.2, -2.08, 0.08],
        'UA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
        'UU': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
        'UC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
        'UG': [-0.01, -1.78, 3.32,  0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26,  0.17],
        'CA': [ 0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
        'CU': [-0.04, -1.5, 3.3,  0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
        'CC': [ -0.01, -1.78, 3.32, 0.3,  8.7, 32, -11.1, -12.2, -29.7, -3.26,  0.49],
        'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
        'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
        'GU': [-0.08, -1.27, 3.18, -0.8,  7,  31, -13.7, -6.6, -18.4, -0.93, 0.44],
        'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
        'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34 ]
    }

    seqLength = len(seq)
    sequence_vector = np.zeros([101, 11])
    k = 2

    # Encode dinucleotide properties for valid positions
    for i in range(0, seqLength - 1):
        sequence_vector[i, 0:11] = phys_dic[seq[i:i + k]]

    return sequence_vector


def nd(seq, seq_length):
    """
    Compute cumulative nucleotide density (ND) per position.

    For each position j, ND is defined as:
        count(seq[j] in seq[0:j+1]) / (j+1)
    i.e., the running fraction of the nucleotide observed at position j.

    Parameters
    ----------
    seq : str
        Input RNA sequence.
    seq_length : int
        Total length used for output (typically 101).

    Returns
    -------
    np.ndarray
        Array of length `seq_length` containing per-position ND values.
    """
    seq = seq.strip()
    nd_list = [None] * seq_length

    for j in range(seq_length):
        if seq[j] == 'A':
            nd_list[j] = round(seq[0:j + 1].count('A') / (j + 1), 3)
        elif seq[j] == 'U':
            nd_list[j] = round(seq[0:j + 1].count('U') / (j + 1), 3)
        elif seq[j] == 'C':
            nd_list[j] = round(seq[0:j + 1].count('C') / (j + 1), 3)
        elif seq[j] == 'G':
            nd_list[j] = round(seq[0:j + 1].count('G') / (j + 1), 3)

    return np.array(nd_list)


def dealwithdata(protein):
    """
    Batch-encode biochemical and k-mer features from paired FASTA files.

    This function reads:
        ./dataset/{protein}_pos.fa
        ./dataset/{protein}_neg.fa
    and encodes each sequence into a fixed-length feature tensor.

    FASTA ASSUMPTION
    ---------------
    The input FASTA is assumed to be organized in blocks of 3 lines per record:
        line i   : header
        line i+1 : sequence
        line i+2 : (optional/unused, e.g., structure or blank)
    This function uses only the sequence line (i+1).

    Output features (concatenated along the last axis)
    --------------------------------------------------
    - 3-dim base encoding with padding indicator (processFastaFile): (101, 3)
    - 1-dim nucleotide density ND (nd): (101, 1)
    - 11-dim dinucleotide physicochemical properties DPCP (dpcp): (101, 11)
    - k-mer frequency tensors for k=1,2,3 (coden): (101, 4 + 16 + 64)

    Final output shape: (N, 101, F), where F = 3 + 1 + 11 + (4+16+64) = 99.

    Parameters
    ----------
    protein : str
        Dataset identifier used to locate the FASTA files.

    Returns
    -------
    np.ndarray
        Feature tensor of shape (N, 101, F) for all sequences from pos and neg files.
        The concatenation order is: [base+pad, ND, DPCP, kmer(1), kmer(2), kmer(3)].
    """
    seq_length = 101
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    dataX = []

    # Process positive and negative sets sequentially
    for label in ['pos', 'neg']:
        fasta_path = f'./dataset/{protein}_{label}.fa'
        with open(fasta_path, 'r') as f:
            lines = f.readlines()

        # Assumes 3 lines per record: header / sequence / (unused)
        for i in range(0, len(lines), 3):
            seq_line = lines[i + 1].strip()

            # Normalize to RNA alphabet; replace ambiguous base 'N' with 'A'
            seq_line = seq_line.replace('T', 'U').replace('N', 'A')

            # Base encoding with padding indicator: (101, 3)
            probMatr = processFastaFile(seq_line)

            # Nucleotide density: (101,)
            probMatr_ND = nd(seq_line, seq_length)

            # DPCP: (101, 11), normalized by sequence length constant
            probMatr_DPCP = dpcp(seq_line) / 101

            # Concatenate base encoding + ND: (101, 4)
            probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))

            # Concatenate + DPCP: (101, 15)
            probMatr_NDPCP = np.column_stack((probMatr_NDCP, probMatr_DPCP))

            # Positional k-mer frequency features
            kmer1 = coden(seq_line, 1, tris1)  # (101, 4)
            kmer2 = coden(seq_line, 2, tris2)  # (101, 16)
            kmer3 = coden(seq_line, 3, tris3)  # (101, 64)
            Kmer = np.hstack((kmer1, kmer2, kmer3))  # (101, 84)

            # Final concatenated feature matrix: (101, 15+84=99)
            Feature_Encoding = np.column_stack((probMatr_NDPCP, Kmer))
            dataX.append(Feature_Encoding)

    dataX = np.stack(dataX, axis=0)
    print(f"[INFO] Encoded {dataX.shape[0]} sequences for {protein}, shape: {dataX.shape}")
    return dataX


def dealwithdata2(seq):
    """
    Encode a single sequence into the same feature representation as `dealwithdata`.

    This is a single-instance version intended for inference or interactive usage.
    It applies the same normalization steps and feature concatenation scheme.

    Parameters
    ----------
    seq : str
        Raw nucleotide sequence (DNA 'T' will be converted to RNA 'U';
        ambiguous 'N' will be replaced with 'A').

    Returns
    -------
    np.ndarray
        Feature tensor of shape (1, 101, F) where F matches `dealwithdata` output.
    """
    line = seq
    seq_length = 101
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    dataX = []
    dataY = []  # Unused placeholder (kept to avoid changing original structure)

    # Normalize to RNA alphabet and remove surrounding whitespace
    line = line.replace('T', 'U').replace('N', 'A').strip()

    # Base encoding + padding: (101, 3)
    probMatr = processFastaFile(line)

    # ND: (101,)
    probMatr_ND = nd(line, seq_length)

    # Combine base + ND: (101, 4)
    probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))

    # DPCP: (101, 11), normalized
    probMatr_DPCP = dpcp(line) / 101

    # Combine (base+ND) + DPCP: (101, 15)
    probMatr_NDPCP = np.column_stack((probMatr_NDCP, probMatr_DPCP))

    # k-mer features: (101, 84)
    kmer1 = coden(line.strip(), 1, tris1)
    kmer2 = coden(line.strip(), 2, tris2)
    kmer3 = coden(line.strip(), 3, tris3)
    Kmer = np.hstack((kmer1, kmer2, kmer3))

    # Final feature: (101, 99)
    Feature_Encoding = np.column_stack((probMatr_NDPCP, Kmer))

    # Keep output batch dimension = 1
    dataX.append(Feature_Encoding.tolist())
    dataX = np.array(dataX)

    return dataX
