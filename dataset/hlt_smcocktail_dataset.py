import torch
from torch.utils.data import Dataset

class HLTSMCocktailDataset(Dataset):
    """
    Loads dict file with keys: pf, obj, label, eventid.
    Creates deterministic train/val split from the TRAIN file.

    Returns (inputs, targets, nuisances) where:
      inputs = (x, mask)
      targets = label (0..4)
      nuisances = AE score scalar per event (if provided) else 0.0

    Split behavior:
      split="train": use (1-val_fraction) of train file
      split="val":   use val_fraction of train file
      split="test":  use full test file (no splitting)
    """

    def __init__(
        self,
        smcocktail_path: str,
        split: str,
        val_fraction: float = 0.1,
        seed: int = 12345,
        ae_scores_path: str = None,
    ):
        assert split in ["train", "val", "test"], f"split must be train/val/test, got {split}"

        d = torch.load(smcocktail_path, map_location="cpu")

        pf = d["pf"]            # [M, 400, 7]
        obj = d["obj"]          # [M, 23, 4]
        y = d["label"].long()   # [M]

        assert pf.dim() == 3, f"pf must be [M,N,F], got {pf.shape}"
        assert obj.dim() == 3, f"obj must be [M,No,Fo], got {obj.shape}"
        assert len(pf) == len(obj) == len(y), "Length mismatch between pf/obj/label"

        # Load nuisance (AE recon score)
        z = None
        if ae_scores_path is not None:
            z = torch.load(ae_scores_path, map_location="cpu")
    
            if z.dim() == 2 and z.shape[1] == 1:
                z = z.view(-1)
                
            assert z.dim() == 1, f"AE scores must be [M] (or [M,1]). Got {z.shape}"
            assert len(z) == len(pf), f"AE scores length {len(z)} != data length {len(pf)}"
            z = z.float()

        # Determine indices for split
        M = len(pf)
        if split == "test":
            idx = torch.arange(M)
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(M, generator=g)
            val_size = int(val_fraction * M)

            if val_size <= 0:
                raise ValueError(f"val_fraction too small; val_size computed as {val_size}")

            if split == "val":
                idx = perm[:val_size]
            else:  # train
                idx = perm[val_size:]

        self.pf = pf[idx]
        self.obj = obj[idx]
        self.y = y[idx]
        self.z = z[idx] if z is not None else None

    def __len__(self):
        return self.pf.shape[0]

    def make_pf_padding_mask(self, x_pf: torch.Tensor) -> torch.Tensor:
        # True where padded tokens are
        return x_pf[..., 0] == 0

    def __getitem__(self, idx):
        x = self.pf[idx]
        mask = self.make_pf_padding_mask(x)
        x = x.clamp(-1e6, 1e6)  # kill fill values (-FLT_MAX, INT_MAX) that cause NaN in transformer
        y = self.y[idx]
        z = self.z[idx] if self.z is not None else torch.tensor(0.0)
        return (x, mask), y, z
