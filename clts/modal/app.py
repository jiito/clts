from pathlib import PosixPath

import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

## APP
app_name = "clts-compute-activations"
app = modal.App(app_name)

gpu = "A100-40GB"

## VOLUME
volume = modal.Volume.from_name(
    "clts-compute-activations-volume", create_if_missing=True
)
volume_path = PosixPath("/vol/data")
model_save_path = volume_path / "models"


## IMAGE
base_image = modal.Image.debian_slim(python_version="3.11").pip_install("pydantic")

torch_image = base_image.pip_install(
    "torch",
    "hf_xet>=1.1.5",
    "numpy",
    "datasets",
    "einops",
    "jaxtyping",
    "h5py",
    "nnsight",
    "tqdm",
)
torch_image = torch_image.pip_install("wandb")

torch_image = torch_image.add_local_python_source("clts")
