from kfp import dsl, client
from kfp.dsl import Dataset, Model, Input, Output, component
from typing import Optional


@component(
    base_image="gitlab-registry.cern.ch/cms-phase2-repr-learning/phase2-samples:latest",
    packages_to_install=[],
)
def pretrain_step(data: Input[Dataset], encoder_out: Output[Model]):
    import torch
    import os
    from train import main as train_main

    output_path = os.path.join(encoder_out.path, "encoder_epoch_400.pt")
    train_main(data_path=data.path)
    torch.save(torch.load("checkpoints/encoder_epoch_400.pt"), output_path)


@component(
    base_image="gitlab-registry.cern.ch/cms-phase2-repr-learning/phase2-samples:latest",
    packages_to_install=[],
)
def posttrain_step(data: Input[Dataset], encoder_in: Input[Model]):
    from post_training import main as posttrain_main

    posttrain_main(data_path=data.path, encoder_path=encoder_in.path)


@dsl.pipeline(name="JEPA-RL Training Pipeline", description="Pretraining + RL posttraining")
def jepa_rl_pipeline(data_path: str):
    import os
    from kfp.dsl import importer

    dataset_importer = importer(
        artifact_class=Dataset,
        artifact_uri="/eos/project/c/cms-l1ml/PhaseII-repr-learning/artifacts",
        reimport=True
    )

    pretrain = pretrain_step(data=dataset_importer.output)
    posttrain_step(data=dataset_importer.output, encoder_in=pretrain.outputs["encoder_out"])


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(jepa_rl_pipeline, "jepa_rl_pipeline.yaml")
