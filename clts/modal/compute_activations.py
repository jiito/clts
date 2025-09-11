from huggingface_hub import HfFolder
from clts.modal.app import app, gpu, HOURS, torch_image, volume_path, volume

from clts.modal.logging import time_dataset_next, timeout_trace
from clts.utils.activations import (
    ActivationCacheConfig,
    get_dataset_and_loader,
    compute_and_save_activations,
    get_nnsight_model,
)


@app.function(
    gpu=gpu,
    timeout=1 * HOURS,
    image=torch_image,
    volumes={volume_path: volume},
)
def compute_activations(cfg: ActivationCacheConfig):
    save_path = volume_path / cfg.save_path / f"{cfg.model_name.replace('/', '_')}.h5"

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: cache the dataset
    dataset, loader = get_dataset_and_loader(cfg)
    model = get_nnsight_model(cfg.model_name)
    print(f"Model is using device: {next(model.parameters()).device}")

    time_dataset_next(loader, tag="train")

    compute_and_save_activations(loader, cfg, model)

    # TODO: log output
    return


@app.local_entrypoint()
def main():
    save_path = volume_path / "activations"
    config = ActivationCacheConfig(
        model_name="roneneldan/TinyStories-1Layer-21M",
        dataset_path="roneneldan/TinyStories",
        save_path=save_path.as_posix(),
        batch_size=1,
        hg_token=HfFolder.get_token(),
    )

    compute_activations.remote(config)
