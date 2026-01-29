# io/dataset_loaders.py
"""
Dataset loaders for VDJdb / McPAS / generic CSV/TSV datasets.
These loaders are small convenience wrappers: they expect local files.
They return:
    sequences: list[str]
    labels: np.array (int or str)
    metadata: dict or structured array
"""
import csv
import numpy as np
import os

def load_generic_tsv(path, seq_col='cdr3', label_col='epitope', delimiter='\t', max_rows=None):
    sequences = []
    labels = []
    metadata = []
    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            seq = row.get(seq_col, None)
            lab = row.get(label_col, None)
            if seq is None:
                continue
            sequences.append(seq)
            labels.append(lab)
            metadata.append(row)
    labels = np.array(labels)
    return sequences, labels, metadata

def load_vdjdb(path, seq_col='cdr3', epitope_col='epitope', max_rows=None):
    # wrapper for typical VDJdb format (may vary). Use generic loader.
    return load_generic_tsv(path, seq_col=seq_col, label_col=epitope_col, delimiter='\t', max_rows=max_rows)

def load_mcpas(path, seq_col='cdr3', label_col='pathogen', delimiter=',', max_rows=None):
    return load_generic_tsv(path, seq_col=seq_col, label_col=label_col, delimiter=delimiter, max_rows=max_rows)

def discover_data_files(data_dir, extensions=['.tsv', '.csv']):
    files = []
    for root, _, fnames in os.walk(data_dir):
        for f in fnames:
            if any(f.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, f))
    return sorted(files)
