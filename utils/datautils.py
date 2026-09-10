"""
General I/O and preprocessing utilities for BRIDGE-style datasets.

This module collects small helper functions used across data preparation and
experiment scripts. The active functions focus on:

- filesystem helpers (creating output folders, listing dataset files)
- simple run bookkeeping (checking whether an expected results file is complete)
- sequence preprocessing (DNA/RNA → one-hot encoding with optional centered padding)
- dataset splitting (simple stratified train/validation split for binary targets)
- lightweight formatting helpers (array → string)

Who this is for
---------------
- Users running BRIDGE-related scripts that need quick dataset utilities.
- Developers who want a single place for shared preprocessing functions.

Dependencies
------------
- Standard library: ``os``, ``sys``, ``copy.deepcopy``
- Third-party: ``numpy``, ``h5py`` (imported here but only required by legacy/extended loaders)

API overview
------------
Filesystem
    ``make_directory(path, foldername, verbose=1)``
        Ensures ``path`` exists, then creates/returns ``os.path.join(path, foldername)``.

    ``get_file_names(dataset_path)``
        Returns all filenames in ``dataset_path`` with extension exactly ``.h5`` (not full paths).

Bookkeeping
    ``finished(path, line_num)``
        Returns True if ``path`` exists and contains exactly ``line_num`` lines.

Formatting
    ``mat2str(m)``
        Converts a 1D/2D numeric array into a comma-separated string with ``'%.3f,'`` formatting
        (note: a trailing comma is included).

Sequence preprocessing
    ``convert_one_hot(sequence, max_length=None)``
        Converts a list of DNA/RNA strings into one-hot arrays with channel order
        ``A, C, G, (U/T)``. Optional centered zero-padding to ``max_length``.

Dataset splitting
    ``split_dataset(data, targets, valid_frac=0.2)``
        Performs a simple stratified split using a binary threshold on targets:
        negatives: ``targets < 0.5``, positives: ``targets >= 0.5``.

Input/Output conventions
------------------------
One-hot encoding
    - Input: ``sequence`` is a ``list[str]`` (each sequence is uppercased internally).
    - Output: ``np.ndarray`` of shape ``(N, 4, L)`` or ``(N, 4, max_length)``.
    - Channel order:
        - channel 0: ``A``
        - channel 1: ``C``
        - channel 2: ``G``
        - channel 3: ``U`` or ``T``

Padding behavior (``convert_one_hot``)
    If ``max_length`` is provided, sequences are centered and padded symmetrically:

    - ``offset1 = (max_length - seq_length) // 2``
    - ``offset2 = max_length - seq_length - offset1``

    The function does not truncate sequences longer than ``max_length``; in that case
    negative offsets would lead to unexpected behavior. Ensure ``max_length >= len(seq)``
    upstream or add explicit truncation.

Split behavior (``split_dataset``)
    - ``data`` is assumed to be indexed by sample along axis 0: ``(N, ...)``.
    - ``targets`` must align with ``data`` along axis 0 (shape ``(N,)`` or ``(N, 1)``).
    - Randomness comes from ``np.random.permutation``; set ``np.random.seed(...)`` upstream
      for reproducibility.

How to use
----------
Convert sequences to one-hot:

.. code-block:: python

    from utils import convert_one_hot
    seqs = ["ACGT", "AUGU"]   # 'U' is treated like 'T' in the 4th channel
    X = convert_one_hot(seqs, max_length=10)   # shape (2, 4, 10)

Split into train/test:

.. code-block:: python

    import numpy as np
    from utils import split_dataset

    X = np.random.randn(100, 4, 101)
    y = (np.random.rand(100, 1) > 0.5).astype(np.float32)

    (X_tr, y_tr), (X_te, y_te) = split_dataset(X, y, valid_frac=0.2)

Notes and caveats
-----------------
- Character handling in ``convert_one_hot``:
  Characters other than ``A/C/G/U/T`` are silently ignored (left as all-zeros). If you need
  explicit handling of ``N`` or other IUPAC codes, extend the implementation.
"""

from __future__ import absolute_import  # Python 2/3 compatibility shim (legacy; harmless under Python 3)
from __future__ import division  # ditto: forces true division under old Python 2 semantics
from __future__ import print_function  # ditto: enables print() as a function

import os, sys, h5py  # os/sys: filesystem ops; h5py: legacy HDF5 loaders (only used by the commented-out functions below)
import numpy as np  # array math for one-hot encoding and dataset splitting
from copy import deepcopy  # imported for legacy callers; unused by the active functions in this file


def make_directory(path, foldername, verbose=1):
    """Make a directory"""

    if not os.path.isdir(path):  # only create the parent if it doesn't already exist
        os.mkdir(path)  # create the parent directory
        print("making directory: " + path)  # log the creation

    outdir = os.path.join(path, foldername)  # target subdirectory path
    if not os.path.isdir(outdir):  # only create the subdirectory if it doesn't already exist
        os.mkdir(outdir)  # create the subdirectory
        print("making directory: " + outdir)  # log the creation
    return outdir  # full path to the (now guaranteed to exist) subdirectory


def finished(path, line_num):
    """Check whether a text results file contains an expected number of lines.

    Args:
        path (str):
            Path to the results file.
        line_num (int):
            Expected total number of lines in the file.

    Returns:
        bool:
            True if file exists and line count matches `line_num`, else False.
    """

    if os.path.exists(path):  # only a file that exists can be "finished"
        with open(path, "r") as f:  # open the results file for reading
            if line_num == len(f.readlines()):  # compare actual line count to the expected count
                return True  # file has exactly the expected number of lines: run is complete
            else:
                return False  # line count mismatch: run is incomplete or file is corrupted
    else:
        return False  # missing file: run hasn't produced results yet


def get_file_names(dataset_path):
    """
    List all HDF5 filenames in a directory.

    Args:
        dataset_path (str):
            Directory containing dataset files.

    Returns:
        list[str]:
            Filenames (not full paths) whose extension is exactly '.h5'.
    """
    file_names = []  # accumulates matching filenames
    for file_name in os.listdir(dataset_path):  # scan every entry in the directory
        if os.path.splitext(file_name)[1] == '.h5':  # keep only files with a literal '.h5' extension
            file_names.append(file_name)  # record this HDF5 filename
    return file_names  # list of '.h5' filenames (not full paths) found in dataset_path


def md5(string):
    """
    Compute the MD5 hex digest of a UTF-8 string.

    Args:
        string (str):
            Input string to hash.

    Returns:
        str:
            Lowercase MD5 hex digest.
    """
    return hashlib.md5(string.encode('utf-8')).hexdigest()  # encode to bytes, then hash and hex-format (note: `hashlib` is not imported in this module, so calling this will raise NameError)


def mat2str(m):
    """
    Convert a 1D or 2D numeric array to a comma-separated string with 3-decimal formatting.

    Args:
        m (np.ndarray):
            1D array shape (D,) or 2D array shape (R, C).

    Returns:
        str:
            Comma-separated string with each value formatted as '%.3f,'.
            Note the string has a trailing comma.

    Notes:
        - For 2D arrays, rows are flattened in row-major order.
        - Does not insert line breaks between rows.
    """
    string = ""  # accumulator for the output string
    if len(m.shape)==1:  # 1D array: emit one value per element
        for j in range(m.shape[0]):  # iterate over the single axis
            string+= "%.3f," % m[j]  # append this value formatted to 3 decimals, with trailing comma
    else:  # 2D array: emit values in row-major order
        for i in range(m.shape[0]):  # iterate over rows
            for j in range(m.shape[1]):  # iterate over columns within the row
                string+= "%.3f," % m[i,j]  # append this cell's value formatted to 3 decimals, with trailing comma
    return string  # comma-separated string of all values (with a trailing comma)
    
    
def convert_one_hot(sequence, max_length=None):
    """
    Convert DNA/RNA sequences to one-hot encoding with optional center padding.

    Encoding:
        Channel order: A, C, G, (U/T)
        - 'A' -> channel 0
        - 'C' -> channel 1
        - 'G' -> channel 2
        - 'U' or 'T' -> channel 3

    Args:
        sequence (list[str]):
            List of sequence strings. Characters are uppercased internally.
        max_length (int, optional):
            If provided, sequences are zero-padded (centered) to this length.
            Padding is applied symmetrically as:
                offset1 = (max_length - seq_length) // 2
                offset2 = max_length - seq_length - offset1
            The resulting shape becomes (N, 4, max_length).

    Returns:
        np.ndarray:
            One-hot encoded array of shape (N, 4, L) or (N, 4, max_length),
            dtype float64 by default due to np.zeros usage.

    Notes:
        - Characters other than A/C/G/U/T are ignored (remain all-zeros at that position).
          If you want explicit handling of 'N' etc., add it upstream.
    """
    one_hot_seq = []  # accumulates one (4, L) or (4, max_length) array per input sequence
    for seq in sequence:  # process each sequence independently
        seq = seq.upper()  # normalize case so matching below is case-insensitive
        seq_length = len(seq)  # this sequence's raw (unpadded) length
        one_hot = np.zeros((4,seq_length))  # preallocate the 4-channel one-hot array for this sequence
        index = [j for j in range(seq_length) if seq[j] == 'A']  # positions where the base is adenine
        one_hot[0,index] = 1  # set the A channel at those positions
        index = [j for j in range(seq_length) if seq[j] == 'C']  # positions where the base is cytosine
        one_hot[1,index] = 1  # set the C channel at those positions
        index = [j for j in range(seq_length) if seq[j] == 'G']  # positions where the base is guanine
        one_hot[2,index] = 1  # set the G channel at those positions
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]  # positions where the base is uracil (RNA) or thymine (DNA)
        one_hot[3,index] = 1  # set the U/T channel at those positions

        # handle boundary conditions with zero-padding
        if max_length:  # only pad if a target length was requested
            offset1 = int((max_length - seq_length)/2)  # zero-padding added before the sequence (centers it)
            offset2 = max_length - seq_length - offset1  # zero-padding added after the sequence (absorbs any rounding remainder)

            if offset1:  # skip if no left padding is needed
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])  # prepend left padding along the length axis
            if offset2:  # skip if no right padding is needed
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])  # append right padding along the length axis

        one_hot_seq.append(one_hot)  # keep this sequence's (padded) one-hot array

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)  # stack all sequences into a single (N, 4, L) array

    return one_hot_seq  # one-hot encoded batch, channel order A, C, G, U/T


def split_dataset(data, targets, valid_frac=0.2):
    """
    Stratified split of a dataset into train/test partitions by a binary threshold.

    This function defines:
        negatives: targets < 0.5
        positives: targets >= 0.5
    and then samples approximately `valid_frac` from each group into the test set.

    Args:
        data (np.ndarray):
            Input samples array. First dimension is assumed to be sample axis (N, ...).
        targets (np.ndarray):
            Target values aligned with `data` along the first axis.
            Can be shape (N,) or (N,1) as long as comparisons and indexing work.
        valid_frac (float, optional):
            Fraction of each class to allocate to the validation/test split. Default: 0.2.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
            (train, test) where:
              train = (X_train, Y_train)
              test  = (X_test,  Y_test)

    Notes:
        - Within each class, indices are randomly permuted via np.random.permutation.
        - The returned train/test are concatenations of positives then negatives (as implemented).
    """    
    ind0 = np.where(targets<0.5)[0]  # indices of negative-class samples
    ind1 = np.where(targets>=0.5)[0]  # indices of positive-class samples

    n_neg = int(len(ind0)*valid_frac)  # number of negatives to hold out for the test split
    n_pos = int(len(ind1)*valid_frac)  # number of positives to hold out for the test split

    shuf_neg = np.random.permutation(len(ind0))  # random ordering over the negative-class indices
    shuf_pos = np.random.permutation(len(ind1))  # random ordering over the positive-class indices

    X_train = np.concatenate((data[ind1[shuf_pos[n_pos:]]], data[ind0[shuf_neg[n_neg:]]]))  # remaining (non-held-out) positives + negatives as training features
    Y_train = np.concatenate((targets[ind1[shuf_pos[n_pos:]]], targets[ind0[shuf_neg[n_neg:]]]))  # matching training targets, same ordering
    train = (X_train, Y_train)  # bundle training features and targets

    X_test = np.concatenate((data[ind1[shuf_pos[:n_pos]]], data[ind0[shuf_neg[:n_neg]]]))  # held-out positives + negatives as test features
    Y_test = np.concatenate((targets[ind1[shuf_pos[:n_pos]]], targets[ind0[shuf_neg[:n_neg]]]))  # matching test targets, same ordering
    test = (X_test, Y_test)  # bundle test features and targets

    return train, test  # (train, test) tuples, each (X, Y), stratified by the 0.5 target threshold


# def rescale(vec, thr=0.0):
#     ind0 = np.where(vec>=thr)[0]
#     u_norm = 0.5 * (vec[ind0]-thr)/(vec[ind0].max()) + 0.5
#     ind2 = np.where(vec<0)[0]
#     vec_norm = vec.copy()
#     vec_norm[ind0] = u_norm
#     vec_norm[ind2] = 0.0
#     return vec_norm

    
# def decodeDNA(m):
#     na=["A","C","G","U"]
#     var,inds=np.where(m==1)
#     seq=""
#     for i in inds:
#         seq=seq+na[i]
#     return seq


# def str_onehot(vec):
#     thr=0.15
#     mask_str = np.zeros((2,vec.shape[-1]))
#     ind =np.where(vec >= thr)[1]
#     mask_str[1,ind]=1
#     ind =np.where(vec < thr)[1]
#     mask_str[0,ind]=1
#     ind =np.where(vec == -1)[1]
#     mask_str[0,ind]=0.5
#     mask_str[1,ind]=0.5
#     return mask_str


# def convert_cat_one_hot(targets):
#     """convert DNA/RNA sequences to a one-hot representation"""
#     t_length = len(targets)
#     cat_num  = len(np.unique(targets))
#     one_hot = np.zeros((t_length, cat_num))
#     for i in range(cat_num):
#         index = np.where(targets==i)[0]
#         one_hot[index,i]= 1
#     return one_hot


# def seq_mutate(seq):
#     mut_seq = []
#     for i in range(len(seq)):
#         if seq[i] == "A" :
#             mut_seq.extend([seq[0:i] + "C" + seq[(i+1):], seq[0:i] + "G" + seq[i+1:], seq[0:i] + "T" + seq[i+1:]])
#         elif seq[i] == "C" :
#             mut_seq.extend([seq[0:i] + "A" + seq[i+1:], seq[0:i] + "G" + seq[i+1:], seq[0:i] + "T" + seq[i+1:]])
#         elif seq[i] == "G" :
#             mut_seq.extend([seq[0:i] + "A" + seq[i+1:], seq[0:i] + "C" + seq[i+1:], seq[0:i] + "T" + seq[i+1:]])
#         else:
#             mut_seq.extend([seq[0:i] + "A" + seq[i+1:], seq[0:i] + "C" + seq[i+1:], seq[0:i] + "G" + seq[i+1:]])
#     return mut_seq


# def load_dataset_hdf5(file_path, ss_type='seq'):

#     def prepare_data(train, ss_type=None):
#         if ss_type == 'struct':
#             structure = train['inputs'][:,:,:,4:9]
#             paired = np.expand_dims(structure[:,:,:,0], axis=3)
#             train['inputs']  = paired
#             return train

#         seq = train['inputs'][:,:,:,:4]

#         if ss_type == 'pu':
#             structure = train['inputs'][:,:,:,4:9]
#             paired = np.expand_dims(structure[:,:,:,0], axis=3)

#             if structure.shape[-1]>3:
#                 unpaired = np.expand_dims(np.sum(structure[:,:,:,1:], axis=3), axis=3)
#                 seq = np.concatenate([seq, paired, unpaired], axis=3)
#             elif structure.shape[-1]==1:
#                 seq = np.concatenate([seq, paired], axis=3)
#             elif structure.shape[-1]==2:
#                 unpaired = np.expand_dims(structure[:,:,:,1], axis=3)
#                 seq = np.concatenate([seq, paired, unpaired], axis=3)
#             elif structure.shape[-1]==3:
#                 unpaired = np.expand_dims(structure[:,:,:,1], axis=3)
#                 other = np.expand_dims(structure[:,:,:,2], axis=3)
#                 seq = np.concatenate([seq, paired, unpaired, other], axis=3)
#         elif ss_type == 'p':
#             structure = train['inputs'][:,:,:,4:9]
#             paired = np.expand_dims(structure[:,:,:,0], axis=3)
#             seq = np.concatenate([seq, paired], axis=3)
#         elif ss_type == 'struct':
#             structure = train['inputs'][:,:,:,4:9]
#             paired = np.expand_dims(structure[:,:,:,0], axis=3)
#             HIME = structure[:,:,:,1:]
#             seq = np.concatenate([seq, paired, HIME], axis=3)
#         train['inputs']  = seq
#         return train

#     # open dataset
#     with h5py.File(file_path, 'r') as f:
#         # load set A data
#         X_train = np.array(f['X_train'])
#         Y_train = np.array(f['Y_train'])
#         X_test  = np.array(f['X_test'])
#         Y_test  = np.array(f['Y_test'])

#     # expand dims of targets
#     if len(Y_train.shape) == 1:
#         Y_train = np.expand_dims(Y_train, axis=1)
#         Y_test  = np.expand_dims(Y_test, axis=1)

#     # add another dimension to make a 4d tensor
#     X_train = np.expand_dims(X_train, axis=3).transpose([0, 2, 3, 1])
#     X_test  = np.expand_dims(X_test,  axis=3).transpose([0, 2, 3, 1])
    
#     # dictionary for each dataset
#     train = {'inputs': X_train, 'targets': Y_train}
#     test  = {'inputs': X_test, 'targets': Y_test}
    

#     # parse secondary structure profiles
#     train = prepare_data(train, ss_type)
#     test  = prepare_data(test, ss_type)

#     print("train:",train['inputs'].shape)
#     print("test:",test['inputs'].shape)

#     return train, test


# def process_data(train, test, method='log_norm'):
#     """get the results for a single experiment specified by rbp_index.
#     Then, preprocess the binding affinity intensities according to method.
#     method:
#         clip_norm - clip datapoints larger than 4 standard deviations from the mean
#         log_norm - log transcormation
#         both - perform clip and log normalization as separate targets (expands dimensions of targets)
#     """

#     def normalize_data(data, method):
#         if method == 'standard':
#             MIN = np.min(data)
#             data = np.log(data-MIN+1)
#             sigma = np.mean(data)
#             data_norm = (data)/sigma
#             params = sigma
#         if method == 'clip_norm':
#             # standard-normal transformation
#             significance = 4
#             std = np.std(data)
#             index = np.where(data > std*significance)[0]
#             data[index] = std*significance
#             mu = np.mean(data)
#             sigma = np.std(data)
#             data_norm = (data-mu)/sigma
#             params = [mu, sigma]

#         elif method == 'log_norm':
#             # log-standard-normal transformation
#             MIN = np.min(data)
#             data = np.log(data-MIN+1)
#             mu = np.mean(data)
#             sigma = np.std(data)
#             data_norm = (data-mu)/sigma
#             params = [MIN, mu, sigma]

#         elif method == 'both':
#             data_norm1, params = normalize_data(data, 'clip_norm')
#             data_norm2, params = normalize_data(data, 'log_norm')
#             data_norm = np.hstack([data_norm1, data_norm2])
#         return data_norm, params


#     # get binding affinities for a given rbp experiment
#     Y_train = train['targets']
#     Y_test = test['targets']
#     #import pdb; pdb.set_trace()

#     if len(Y_train.shape)==1:
#         # filter NaN
#         train_index = np.where(np.isnan(Y_train) == False)[0]
#         test_index = np.where(np.isnan(Y_test) == False)[0]
#         Y_train = Y_train[train_index]
#         Y_test = Y_test[test_index]
#         X_train = train['inputs'][train_index]
#         X_test = test['inputs'][test_index]
#     else:
#         X_train = train['inputs']
#         X_test = test['inputs']

#     # normalize intenensities
#     if method:
#         Y_train, params_train = normalize_data(Y_train, method)
#         Y_test, params_test = normalize_data(Y_test, method)

#     # store sequences and intensities
#     train = {'inputs': X_train, 'targets': Y_train}
#     test = {'inputs': X_test, 'targets': Y_test}

#     return train, test


# def down_negative_samples(train, test, ratio=0.0):
#     """get the results for a single experiment specified by rbp_index.
#     Then, preprocess the binding affinity intensities according to method.
#     method:
#         clip_norm - clip datapoints larger than 4 standard deviations from the mean
#         log_norm - log transcormation
#         both - perform clip and log normalization as separate targets (expands dimensions of targets)
#     """
#     if ratio==0.0:
#         print("No negative down-sampling ratio.")
#         return train, test

#     X_train = train['inputs']
#     X_test  = test['inputs']

#     Y_train = train['targets']#.astype(np.int32)
#     Y_test  = test['targets']#.astype(np.int32)

#     pos_index_tr = np.where(Y_train==1)[0]
#     pos_index_te = np.where(Y_test==1)[0]

#     neg_index_tr = np.where(Y_train==0)[0]
#     neg_index_te = np.where(Y_test==0)[0]

#     n_down_neg_tr = int(ratio * (len(Y_train) - len(neg_index_tr)))
#     n_down_neg_te = int(ratio * (len(Y_test) -  len(neg_index_te)))

#     dw_neg_index_tr = np.random.choice(neg_index_tr, size=n_down_neg_tr)
#     dw_neg_index_te = np.random.choice(neg_index_te, size=n_down_neg_te)

#     pos_neg_tr =np.concatenate((dw_neg_index_tr,    pos_index_tr))
#     pos_neg_te =np.concatenate((dw_neg_index_te,    pos_index_te))

#     train = {'inputs': X_train[pos_neg_tr], 'targets': Y_train[pos_neg_tr]}
#     test = {'inputs': X_test[pos_neg_te], 'targets': Y_test[pos_neg_te]}

#     return train, test


# def load_testset_txt_only_seq(filepath, test, return_trans_id=False, seq_length=101):
#     print("Reading inference file(only seq):", filepath)
#     if os.path.exists(filepath+"_test.h5"):
#         print("loading from h5.")        
#         with h5py.File(filepath+"_test.h5", 'r') as f:
#             # load set A data
#             test['inputs']  = f['inputs']
#             test['targets'] = f['targets']
     
        
#         if return_trans_id:
#             blob = np.load(filepath+"_tran.npz")
#             trans_ids = blob['trans_ids']
#             return test, trans_ids
#         else:
#             return test

#     seqs = []
#     trans_ids = []
#     with open(filepath,"r") as f:
#         for line in f.readlines():
#             line=line.strip('\n').split('\t')
#             if len(line[2])!=seq_length:
#                 continue
#             trans_ids.append(line[0])
#             seqs.append(line[1])
#     print("Converting.")        
#     input = convert_one_hot(seqs, seq_length)
#     print("Converted.")        

#     inputs = np.expand_dims(input, axis=3).transpose([0, 2, 3, 1])
#     targets = np.ones((inputs.shape[0],1))
#     targets[inputs.shape[0]-1]=0

#     test['inputs'] =inputs
#     test['targets'] =targets
    
#     print("Saving into h5.")
#     with h5py.File(filepath+"_test.h5", "w") as f:
#         dset = f.create_dataset("inputs", data=inputs, compression="gzip")
#         dset = f.create_dataset("targets", data=targets, compression="gzip")
#     print("Saved.")

#     if return_trans_id:
#         trans_ids = np.array(trans_ids)
#         return test, trans_ids
#     else:
#         return test


# def load_testset_txt(filepath, use_structure=True, seq_length=101):
#     test = {}

#     print("Reading inference file:", filepath)
#     if os.path.exists(filepath+"_test.npz"):
#         print("loading from npz.")        
       
#         f = np.load(filepath+"_test.npz", allow_pickle=True)
#         test['inputs']  = f['inputs']
#         test['targets'] = f['targets']

#         return test

#     in_ver = 5
#     seqs = []
#     strs = []
#     with open(filepath,"r") as f:
#         for line in f.readlines():
#             line=line.strip('\n').split('\t')
#             if len(line[2])!=seq_length:
#                 continue
#             seqs.append(line[2])
#             if use_structure:
#                 strs.append(line[3])
#     in_seq = convert_one_hot(seqs, seq_length)
    
#     if use_structure:
#         structure = np.zeros((len(seqs), in_ver-4, seq_length))
#         for i in range(len(seqs)):
#             icshape = strs[i].strip(',').split(',')
#             ti = [float(t) for t in icshape]
#             ti = np.array(ti).reshape(1,-1)
#             structure[i] = np.concatenate([ti], axis=0)
#         input = np.concatenate([in_seq, structure], axis=1)
#     else:
#         input = in_seq

#     inputs = np.expand_dims(input, axis=3).transpose([0, 3, 2, 1])
#     targets = np.ones((in_seq.shape[0],1))

#     targets[in_seq.shape[0]-1]=0

#     test['inputs']  = inputs
#     test['targets'] = targets
#     print("Saving into npz.")
#     np.savez_compressed(filepath+"_test.npz", inputs=inputs, targets=targets)
#     print("Saved.")

#     return test


# def load_testset_txt_mu(filepath, test, seq_length=101):
#     print("Reading test file:", filepath)
#     f_mu = open(filepath,"r")
#     seqs = []
#     strs = []
#     use_pu = True
#     if test['inputs'].shape[-1]==4:
#         use_pu = False
#     nf=0
#     for line in f_mu.readlines():
#         nf+=1
#         line=line.strip('\n').split('\t')
#         if len(line[2])!=seq_length:
#             continue
#         seqs.append(line[2])
#         mut_seq=seq_mutate(line[2])
#         seqs.extend(mut_seq)
#         if use_pu:
#             strs.extend([line[3]] * len(seqs))
#     print("file line num:",nf)
#     print("mut seq num:",len(seqs))
#     in_seq = munge.convert_one_hot(seqs, seq_length)
#     in_ver = 5
#     if use_pu:
#         structure = np.zeros((len(seqs), in_ver-4, seq_length))
#         for i in range(len(seqs)):
#             struct_list = strs[i].strip(',').split(',')
#             ti = np.array([float(t) for t in struct_list]).reshape(1,-1)
#             structure[i] = np.concatenate([ti], axis=0)
#         input = np.concatenate([in_seq, structure], axis=1)
#     else:
#         input = in_seq

#     inputs = np.expand_dims(input, axis=3).transpose([0, 2, 3, 1])
#     targets = np.ones((in_seq.shape[0],1))

#     targets[in_seq.shape[0]-1]=0

#     test['inputs'] =inputs
#     test['targets'] =targets
#     return test