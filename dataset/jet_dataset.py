import os
from collections import Counter
import numpy as np
import multiprocessing
import math
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
from sklearn.model_selection import train_test_split


def create_jet_features_data():
  file = h5py.File('particles.h5', 'r')
  print(file.keys())
  jet_vars = np.array(file['jets']).astype(np.float64)
  jet_spherecity_vals = np.array(file['spherecity_vals']).astype(np.float64)

  #We get Data only from few colums
  data = np.array(file.get('jets')[:,[12, 36, 37, 39, 40, 41, 42, 43, 45]])
  tau21 = (jet_vars[:,5]/jet_vars[:,4])
  tau32 = (jet_vars[:,6]/jet_vars[:,5])

  #Some data cleaning
  mask_1 = tau21<=1.0
  mask_2 = tau32<=1.0
  mask_3 = tau21>=0.0
  mask_4 = tau32>=0.0
  mask = mask_1 * mask_2 * mask_3 * mask_4
  jet_vars = jet_vars[mask]
  data = data[mask]
  data = np.concatenate([
    (jet_vars[:,5]/jet_vars[:,4])[:,np.newaxis],
    (jet_vars[:,6]/jet_vars[:,5])[:,np.newaxis],
    data,
    jet_vars[:,3][:,np.newaxis]],axis=1)

  #Note Jet_vars contains all the colums of data. Spurious correlation is with colum 3 / 48

  #Seprating Jet data in # of quarks it decays to
  three = (jet_vars[:,57]>0) 
  two = (jet_vars[:,56]>0) + (jet_vars[:,55]>0)
  one = (jet_vars[:,53]>0) + (jet_vars[:,54]>0)
  three_prong = data[three]
  two_prong = data[two]
  qcd = data[one]
  print("one, two, three", qcd.shape, two_prong.shape, three_prong.shape)
  
  # save OOD data
  np.save("jet_features_ood.npy", three_prong)

  # split in-dist into train, val, test
  features = np.vstack([qcd, two_prong])
  labels = np.concatenate([np.zeros(len(qcd)), np.ones(len(two_prong))])

  X_tr, X_ts, y_tr, y_ts = train_test_split(features, labels, test_size=.2, random_state=0)
  X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=.2, random_state=0)
  np.savez("jet_features_id.npz", X_train=X_tr, X_val=X_va, X_test=X_ts, y_train=y_tr, y_val=y_va, y_test=y_ts)


class JetFeaturesDataset(Dataset):
  """
  High-level jet features (only 9 of them).
  QCD and Two-prong are in-distribution, Three-prong is OOD.

  """
  def __init__(self, args, split):
    super(JetFeaturesDataset, self).__init__()
    self.split = split
    if args.undersample:
      raise NotImplementedError("undersample not supported for jet features")
    self.data_label_correlation = args.data_label_correlation
    if split == "ood":
      if not os.path.exists("jet_features_ood.npy"):
        create_jet_features_data()
      data = np.load("jet_features_ood.npy")
      self.features = data.astype(np.float32)
      self.labels = np.zeros(len(data)).astype(np.float32)  # placeholder, doesn't mean anything
      self.nuisances = self.labels  # placeholder, doesn't mean anything
    else:
      if not os.path.exists("jet_features_id.npz"):
        create_jet_features_data()
      data = np.load("jet_features_id.npz")
      self.features = data[f"X_{split}"].astype(np.float32)
      self.labels = data[f"y_{split}"].astype(np.float32)
      self.nuisances = self.features[:, -1]

      # bucket mass into groups, only used when exact = 1
      group_counts = Counter()
      for y, z in zip(self.labels, self.nuisances):
        g = math.ceil(z//50)
        group_counts[(y, g)] += 1
      print(group_counts)
      # weights should be inverse of count proportions
      weights = {k: len(self.labels) / v for k, v in group_counts.items()}
      self.weights = {k: v / sum(weights.values()) for k, v in weights.items()}
  
  def __getitem__(self, index):
    return self.features[index], self.labels[index], float(math.ceil(self.nuisances[index]//50))
  
  def __len__(self):
    return len(self.features)
  
  def get_label_prior(self):
    proportion_ones = sum(self.labels)/len(self.labels)
    return {1: proportion_ones, 0: 1 - proportion_ones}
  
  def get_nuisance_prior(self):
    num_examples = len(self.nuisances)
    nuisance_counts = Counter()
    for z in self.nuisances:
      g = math.ceil(z//50)
      nuisance_counts[g] += 1
    nuisance_prior = {k: v/num_examples for k, v in nuisance_counts.items()}
    print(nuisance_prior)
    return nuisance_prior


def get_jet_features_dataloader(args, data_label_correlation, split, root_dir="datasets", **kwargs):
    kwargs = {'pin_memory': False, 'num_workers': multiprocessing.cpu_count(), 'drop_last': False}
    dataset = JetFeaturesDataset(args, split=split)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            **kwargs)
    return dataloader
