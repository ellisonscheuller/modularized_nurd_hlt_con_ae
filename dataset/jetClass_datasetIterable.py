import uproot
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math
import awkward as ak
import multiprocessing
from collections import Counter
import os

def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))

def _pad(a, maxlen, value=0):
    return ak.fill_none(ak.pad_none(a, maxlen, clip=True, axis=1), value)

def build_features_and_labels(tree, transform_features=True):
    # Load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])
    # Compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt'] / a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy'] / a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])
    a['part_d0'] = np.tanh(a['part_d0val'])
    a['part_dz'] = np.tanh(a['part_dzval'])
    
    # Apply standardization
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0
        a['part_d0err'] = _clip(a['part_d0err'], 0, 1)
        a['part_dzerr'] = _clip(a['part_dzerr'], 0, 1)

    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'],
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',
            'part_d0',
            'part_d0err',
            'part_dz',
            'part_dzerr',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)  # Shape: (n_jets, n_features, n_particles)

    label_list = [
        'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
        'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl'
    ]
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)
    
    return out

from torch.utils.data import IterableDataset, DataLoader
import itertools

class JetDatasetIterable(IterableDataset):
    """
    PyTorch IterableDataset for loading jet data from ROOT files.
    """
    def __init__(self, root_files_input, transform_features=True):
        super().__init__()
        self.transform_features = transform_features
        self.root_files_input = root_files_input
        self.files = [os.path.join(root_files_input, f) for f in os.listdir(root_files_input)]
        self.transform_features = transform_features

    def parse_file(self, filepath):
        tree = uproot.open(filepath)['tree']
        table = build_features_and_labels(tree, self.transform_features)
        x_particles = table['pf_features']
        x_jets = table['pf_vectors']
        y = table['label']
        x_points = table['pf_points']
        x_mask = table['pf_mask']

        pf_vectors = x_jets
        px = pf_vectors[:, 0, :]
        py = pf_vectors[:, 1, :]
        pz = pf_vectors[:, 2, :]
        E = pf_vectors[:, 3, :]

        total_px = np.sum(px, axis=1)
        total_py = np.sum(py, axis=1)
        total_pz = np.sum(pz, axis=1)
        total_E = np.sum(E, axis=1)

        mass_squared = total_E**2 - (total_px**2 + total_py**2 + total_pz**2)
        mass_squared = np.maximum(mass_squared, 0)
        masses = np.sqrt(mass_squared)
        nuisances = np.array([math.ceil(n // 50) for n in masses]).astype(np.float32)

        labels_scalar = np.argmax(y, axis=1)

        for i in range(labels_scalar.shape[0]):
            yield (
                x_particles[i],
                x_jets[i],
                x_points[i],
                x_mask[i],
                labels_scalar[i],
                nuisances[i]
            )

    def __iter__(self):
        # Chain results from all files to form an iterable
        return itertools.chain.from_iterable(self.parse_file(f) for f in self.files)

def get_JetClass_iterable_dataloader(root_files, batch_size=32, **kwargs):
    dataset = JetDatasetIterable(root_files_input=root_files, transform_features=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)
    return dataloader

