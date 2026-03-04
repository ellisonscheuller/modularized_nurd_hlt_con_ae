import argparse
import datetime
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import glob
import time
import random

import torch
import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm

from embedding.utils.cfg_handler import data_config, join_remote
from embedding.utils.data_utils import softkill, EPS

uproot.source.xrootd.XRootDSource.timeout = 1800

def expand_input_to_file_paths(file_name: str, sample_dir: Path) -> List[str]:
    """Expand a filename or .txt filelist into a list of paths/XRootD URLs."""
    file_name = str(file_name).strip()

    def norm_one(s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        if s.startswith("root://"):
            # prefer FNAL redirector (more reliable for condor @ LPC)
            s = s.replace("root://cms-xrd-global.cern.ch/", "root://cmsxrootd.fnal.gov/")
            s = s.replace("root://eoscms.cern.ch/", "root://cmsxrootd.fnal.gov/")
            return s
        return os.fspath(sample_dir / s)

    if file_name.endswith(".txt"):
        txt_path = sample_dir / file_name
        if not txt_path.exists():
            raise FileNotFoundError(f"Filelist {txt_path} not found.")
        out = []
        with open(txt_path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                out.append(norm_one(ln))
        return out

    p = sample_dir / file_name
    return [os.fspath(x) for x in glob.glob(os.fspath(p))]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("converterHLT")

def gather_event_ids(tree: uproot.TTree, max_events: int = -1) -> torch.Tensor:
    """Returns (N, 3) int64 tensor: [run, lumi, event]."""
    arr = tree.arrays(["run", "luminosityBlock", "event"], entry_stop=max_events)
    run  = np.asarray(arr["run"], dtype=np.int64)
    lumi = np.asarray(arr["luminosityBlock"], dtype=np.int64)
    evt  = np.asarray(arr["event"], dtype=np.int64)
    out = np.stack([run, lumi, evt], axis=1)
    return torch.from_numpy(out)


def construct_pf_features(branches: ak.Array) -> ak.Array:
    branches["ScoutingParticle_is_pf"] = ak.ones_like(branches["ScoutingParticle_pt"])

    branches["ScoutingMuonNoVtx_dxy"] = branches["ScoutingMuonNoVtx_trk_dxy"]
    branches["ScoutingMuonNoVtx_dxysig"] = (
        branches["ScoutingMuonNoVtx_trk_dxy"] / (branches["ScoutingMuonNoVtx_trk_dxyError"] + EPS)
    )
    branches["ScoutingMuonNoVtx_pdgId"] = branches["ScoutingMuonNoVtx_charge"] * 13
    branches["ScoutingMuonNoVtx_is_pf"] = ak.zeros_like(branches["ScoutingMuonNoVtx_pt"])

    branches["ScoutingElectron_dxy"] = -branches["ScoutingElectron_bestTrack_d0"]
    branches["ScoutingElectron_dxysig"] = ak.zeros_like(branches["ScoutingElectron_pt"])
    branches["ScoutingElectron_pdgId"] = branches["ScoutingElectron_bestTrack_charge"] * 11
    branches["ScoutingElectron_is_pf"] = ak.zeros_like(branches["ScoutingElectron_pt"])

    branches["ScoutingPhoton_dxy"] = ak.zeros_like(branches["ScoutingPhoton_pt"])
    branches["ScoutingPhoton_dxysig"] = ak.zeros_like(branches["ScoutingPhoton_pt"])
    branches["ScoutingPhoton_pdgId"] = ak.ones_like(branches["ScoutingPhoton_pt"]) * 22
    branches["ScoutingPhoton_is_pf"] = ak.zeros_like(branches["ScoutingPhoton_pt"])

    return branches


def gather_pfcands(tree: uproot.TTree, max_events: int = -1) -> ak.Array:
    branches = tree.arrays([
        "ScoutingParticle_pt",
        "ScoutingParticle_eta",
        "ScoutingParticle_phi",
        "ScoutingParticle_dxy",
        "ScoutingParticle_dxysig",
        "ScoutingParticle_pdgId",
        "ScoutingMuonNoVtx_pt",
        "ScoutingMuonNoVtx_eta",
        "ScoutingMuonNoVtx_phi",
        "ScoutingMuonNoVtx_trk_dxy",
        "ScoutingMuonNoVtx_trk_dxyError",
        "ScoutingMuonNoVtx_charge",
        "ScoutingElectron_pt",
        "ScoutingElectron_eta",
        "ScoutingElectron_phi",
        "ScoutingElectron_bestTrack_d0",
        "ScoutingElectron_bestTrack_charge",
        "ScoutingPhoton_pt",
        "ScoutingPhoton_eta",
        "ScoutingPhoton_phi",
    ], entry_stop=max_events)

    branches = construct_pf_features(branches)

    branch_prefixes = [
        "ScoutingParticle_",
        "ScoutingMuonNoVtx_",
        "ScoutingElectron_",
        "ScoutingPhoton_",
    ]
    feature_names = ["pt", "eta", "phi", "dxy", "dxysig", "is_pf", "pdgId"]

    combined = ak.concatenate([
        ak.zip({ftr_name: branches[prefix + ftr_name] for ftr_name in feature_names})
        for prefix in branch_prefixes
    ], axis=1)

    return combined


def process_pfcands(
    combined: ak.Array,
    n_objects: int = 200,
    sk_cell_size: Optional[float] = None,
    sort_by_pt: bool = False,
) -> torch.Tensor:
    """Returns (N, n_objects, n_pf_features) float32 tensor."""
    if sort_by_pt:
        combined = combined[ak.argsort(combined.pt, axis=1, ascending=False)]

    counts = ak.num(combined.pt, axis=1)
    clipped_events = counts > n_objects
    if ak.any(clipped_events):
        lost = int(ak.sum(counts[clipped_events] - n_objects))
        logger.warning(f"{int(ak.sum(clipped_events))} events clipped; {lost} PF cands lost. Increase n_objects.")

    padded = ak.pad_none(combined, n_objects, axis=1, clip=True)
    array = np.stack([ak.to_numpy(padded[field]) for field in padded.fields], axis=-1)
    tensor = torch.tensor(array, dtype=torch.float32)

    if sk_cell_size is not None:
        tensor = softkill(tensor, cell_size=sk_cell_size)

    return tensor


def gather_objects_for_ae(tree: uproot.TTree, max_events: int = -1) -> ak.Array:
    def _topk_pad(pt, eta, phi, type_id: int, k: int):
        order = ak.argsort(pt, axis=1, ascending=False)
        pt_s = pt[order]
        eta_s = eta[order]
        phi_s = phi[order]

        pt_p = ak.pad_none(pt_s,  k, axis=1, clip=True)
        eta_p = ak.pad_none(eta_s, k, axis=1, clip=True)
        phi_p = ak.pad_none(phi_s, k, axis=1, clip=True)
        typ = ak.ones_like(pt_p) * type_id

        return ak.zip({"pt": pt_p, "eta": eta_p, "phi": phi_p, "type": typ})

    # Jets (10)
    jet_pt  = tree["ScoutingPFJetRecluster_pt"].array(entry_stop=max_events)
    jet_eta = tree["ScoutingPFJetRecluster_eta"].array(entry_stop=max_events)
    jet_phi = tree["ScoutingPFJetRecluster_phi"].array(entry_stop=max_events)
    jets = _topk_pad(jet_pt, jet_eta, jet_phi, type_id=0, k=10)

    # Muons (4)
    mu_pt  = tree["ScoutingMuonVtx_pt"].array(entry_stop=max_events)
    mu_eta = tree["ScoutingMuonVtx_eta"].array(entry_stop=max_events)
    mu_phi = tree["ScoutingMuonVtx_phi"].array(entry_stop=max_events)
    muons = _topk_pad(mu_pt, mu_eta, mu_phi, type_id=1, k=4)

    # Electrons (4)
    e_pt  = tree["ScoutingElectron_pt"].array(entry_stop=max_events)
    e_eta = tree["ScoutingElectron_eta"].array(entry_stop=max_events)
    e_phi = tree["ScoutingElectron_phi"].array(entry_stop=max_events)
    electrons = _topk_pad(e_pt, e_eta, e_phi, type_id=2, k=4)

    # Photons (4)
    g_pt  = tree["ScoutingPhoton_pt"].array(entry_stop=max_events)
    g_eta = tree["ScoutingPhoton_eta"].array(entry_stop=max_events)
    g_phi = tree["ScoutingPhoton_phi"].array(entry_stop=max_events)
    photons = _topk_pad(g_pt, g_eta, g_phi, type_id=3, k=4)

    # MET (1)
    met_pt  = tree["ScoutingMET_pt"].array(entry_stop=max_events)
    met_phi = tree["ScoutingMET_phi"].array(entry_stop=max_events)
    met_eta = ak.zeros_like(met_pt)

    met_pt  = met_pt[:, np.newaxis]
    met_eta = met_eta[:, np.newaxis]
    met_phi = met_phi[:, np.newaxis]

    met = ak.zip({
        "pt": met_pt,
        "eta": met_eta,
        "phi": met_phi,
        "type": ak.ones_like(met_pt) * 4,
    })

    return ak.concatenate([jets, muons, electrons, photons, met], axis=1)


def process_objects_for_ae(combined: ak.Array) -> torch.Tensor:
    """Returns (N, 23, 4) float32 tensor: (pt, eta, phi, type)."""
    array = np.stack(
        [
            ak.to_numpy(combined["pt"]),
            ak.to_numpy(combined["eta"]),
            ak.to_numpy(combined["phi"]),
            ak.to_numpy(combined["type"]),
        ],
        axis=-1,
    )
    return torch.tensor(array, dtype=torch.float32)


def save_dataset(
    out_file: Path,
    pf: torch.Tensor,
    y: torch.Tensor,
    obj: Optional[torch.Tensor],
    eventid: Optional[torch.Tensor],
) -> None:
    payload: Dict[str, Any] = {"pf": pf, "label": y}
    if obj is not None:
        payload["obj"] = obj
    if eventid is not None:
        payload["eventid"] = eventid
    torch.save(payload, os.fspath(out_file))


def main(cfg: data_config, overwrite: bool = False):
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    sample_dir = Path(cfg["sample_dir"]).expanduser()
    redir = cfg.get("redir", "")

    n_pf = cfg.get("n_objects", 500)
    nevents_per_class = cfg.get("nevents_per_class", -1)

    pfcands = cfg.get("pfcands", True)
    sort_by_pt = cfg.get("sort_by_pt", True)
    sk_cell_size = cfg.get("sk_spacing", None)

    split = cfg.get("split", None)
    store_by_class = cfg.get("store_by_class", False)
    also_save_objects = cfg.get("also_save_objects", True)

    if store_by_class:
        raise ValueError("store_by_class is not supported in the new combined-file format. Use split or single output.")

    file_label_tuples = cfg.get_file_label_map()

    pf_chunks: List[torch.Tensor] = []
    y_chunks:  List[torch.Tensor] = []
    obj_chunks: List[torch.Tensor] = []
    id_chunks:  List[torch.Tensor] = []

    for file_name, label in tqdm(file_label_tuples, desc="Processing files"):
        file_paths = expand_input_to_file_paths(file_name, sample_dir)
        n_events_left = nevents_per_class

        for src in file_paths:
            if redir and (not src.startswith("root://")):
                src = join_remote(redir, src)

            def open_events_tree(src: str, max_tries: int = 5, base_sleep: float = 10.0):
                last_err = None
                for i in range(max_tries):
                    try:
                        return uproot.open(src, timeout=uproot.source.xrootd.XRootDSource.timeout)["Events"]
                    except Exception as e:
                        last_err = e
                        sleep_s = base_sleep * (2 ** i) + random.uniform(0, 3)
                        logger.warning(f"Failed to open {src} (try {i+1}/{max_tries}): {e}. Sleeping {sleep_s:.1f}s")
                        time.sleep(sleep_s)
                raise last_err
            logger.info(f"Opening: {src}")
            tree = open_events_tree(src)

            ids = gather_event_ids(tree, max_events=n_events_left) if also_save_objects else None

            if not pfcands:
                raise ValueError("pfcands=False is not supported.")

            pf_evt = process_pfcands(
                gather_pfcands(tree, max_events=n_events_left),
                n_objects=n_pf,
                sk_cell_size=sk_cell_size,
                sort_by_pt=sort_by_pt,
            )

            y_evt = torch.full((pf_evt.shape[0],), int(label), dtype=torch.long)

            if also_save_objects:
                obj_evt = process_objects_for_ae(
                    gather_objects_for_ae(tree, max_events=n_events_left)
                )
                if obj_evt.shape[0] != pf_evt.shape[0]:
                    raise RuntimeError(f"Event mismatch label {label}: pf {pf_evt.shape[0]} vs obj {obj_evt.shape[0]}")
                if ids is not None and ids.shape[0] != pf_evt.shape[0]:
                    raise RuntimeError(f"EventID mismatch label {label}: ids {ids.shape[0]} vs pf {pf_evt.shape[0]}")
                obj_chunks.append(obj_evt)
                id_chunks.append(ids)

            pf_chunks.append(pf_evt)
            y_chunks.append(y_evt)

            n_events_left -= pf_evt.shape[0]
            if n_events_left <= 0:
                break

    pf_full = torch.cat(pf_chunks, dim=0)
    pf_full = torch.nan_to_num(pf_full, nan=0.0, posinf=0.0, neginf=0.0)
    y_full  = torch.cat(y_chunks, dim=0)

    obj_full = None
    id_full = None
    if also_save_objects:
        obj_full = torch.cat(obj_chunks, dim=0)
        obj_full = torch.nan_to_num(obj_full, nan=0.0, posinf=0.0, neginf=0.0)
        id_full  = torch.cat(id_chunks, dim=0)

    perm = torch.randperm(pf_full.shape[0])
    pf_full = pf_full[perm]
    y_full  = y_full[perm]
    if also_save_objects:
        obj_full = obj_full[perm]
        id_full  = id_full[perm]

    output_prefix = cfg.get_ds_name() or "embedding_hlt_ssl"
    out_path = Path(cfg.get("out_path", "./")).expanduser()
    os.makedirs(out_path, exist_ok=True)

    if split is not None:
        split_idx = int(float(split) * pf_full.shape[0])

        train_file = out_path / f"{output_prefix}_train.pt"
        test_file  = out_path / f"{output_prefix}_test.pt"
        if (train_file.exists() or test_file.exists()) and not overwrite:
            raise FileExistsError(f"Output exists in {os.fspath(out_path)}. Use --overwrite.")

        save_dataset(train_file, pf_full[:split_idx], y_full[:split_idx],
                     None if obj_full is None else obj_full[:split_idx],
                     None if id_full is None else id_full[:split_idx])
        save_dataset(test_file, pf_full[split_idx:], y_full[split_idx:],
                     None if obj_full is None else obj_full[split_idx:],
                     None if id_full is None else id_full[split_idx:])

        logger.info(f"Saved: {train_file}")
        logger.info(f"Saved: {test_file}")
    else:
        out_file = out_path / f"{output_prefix}.pt"
        if out_file.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {os.fspath(out_file)}. Use --overwrite.")
        save_dataset(out_file, pf_full, y_full, obj_full, id_full)
        logger.info(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config .yaml file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    cfg = data_config(args.config)

    os.makedirs("logs", exist_ok=True)

    log_filename = f"logs/converterHLT_{cfg.get_ds_name()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting conversion with config: {args.config}")
    main(cfg, overwrite=args.overwrite)
