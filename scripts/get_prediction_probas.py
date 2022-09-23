import torch
import wandb
import numpy as np
import argparse
from collections import defaultdict
from attrdict import AttrDict
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
sys.path.append('/misc/vlgscratch5/RanganathGroup/lily/physics_ood/nuisance-aware-ood-detection/')
from models.mlp import get_model_mlp
from utils.eval_utils import get_dataloaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--sweep_name', required=True, default="lily/nurd-ood-jets/sweeps/c29cqb3k", type=str)
    args = parser.parse_args()


    # get data
    _, testloaderIn, testloaderOut = get_dataloaders(
        AttrDict({"in_dataset": "jet_features", "out_dataset": "jet_features-other",
        "batch_size": 256, "data_label_correlation": 0.5, "undersample": False})  # data_label_correlation is just a dummy variable, no effect
    )

    api = wandb.Api()
    sweep = api.sweep(f"{args.sweep_name}")
    wandb_dir = "/misc/vlgscratch5/RanganathGroup/lily/physics_ood/nuisance-aware-ood-detection/wandb/"
    for run in sweep.runs:
        config_dict = run.config
        config_dict["exp_name"] = run.id
        config = AttrDict(config_dict)
        model = get_model_mlp(config)
        inputs_dict = defaultdict(dict)

        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        for split, loader in zip(["id", "ood"], [testloaderIn, testloaderOut]):
            for j, data in enumerate(loader):
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs_cuda = inputs.to(device)

                _, outputs = model(inputs_cuda)

                nnOutputs = outputs.data.cpu().numpy().tolist()
                inputs = inputs.data.cpu().numpy().tolist()
                labels = labels.data.cpu().numpy().tolist()
                for i in range(len(inputs)):
                    if (split, j, i) in inputs_dict:
                        assert inputs_dict[(split, j, i)]["input"] == inputs[i]
                        assert inputs_dict[(split, j, i)]["label"] == labels[i]
                    else:
                        inputs_dict[(split, j, i)]["input"] = inputs[i]
                        inputs_dict[(split, j, i)]["label"] = labels[i]
                        inputs_dict[(split, j, i)]["preds"] = nnOutputs[i]
    
        import json
        with open(f'results/{run.id}_{config.reweight}.jsonl', 'w') as outfile:
            for key in inputs_dict:
                single_result = {}
                single_result["input"] = inputs_dict[key]["input"]
                single_result["label"] = inputs_dict[key]["label"]
                single_result["preds"] = inputs_dict[key]["preds"]
                outfile.write(json.dumps(single_result) + '\n')

        

