# CLTs (Cross‑Layer Transcoders)

This repository is a practice implementation of Cross‑Layer Transcoders (CLTs). It is actively in progress.

This model is based off the work of Anthropic here: https://transformer-circuits.pub/2024/crosscoders/index.html

CLTs were designed to overcome the limitations in SAEs by approximating + sparsifying the MLP, allowing us to construct features that are linear combinations of eachother. This helps us work with cross-layer superposition and greatly simplifies circuit tracing.

I’m following the CLT design in this reference implementation as a learning/example guide: [`crosslayer-transcoder/model/clt.py`](https://github.com/Goreg12345/crosslayer-transcoder/blob/master/crosslayer_transcoder/model/clt.py).

## Status

- Work in progress: expect breaking changes and rapid iteration.

## Where things live

- Model setup

  - `clts/models/clt.py`: Defines the core `CrossLayerTranscoder` (encoder, JumpReLU activation, decoder, standardization) and parameter initialization.
  - Related components: `clts/models/jump_relu.py`, `clts/models/standardizer.py`.

- Activation calculation and caching (Modal)

  - `clts/modal/compute_activations.py`: Modal function to compute and persist activations to an `.h5` cache.
  - `clts/modal/app.py`: Modal app/image/volume configuration (GPU, dependencies, and a persistent `modal.Volume`). Activations are stored under `/vol/data/activations/` inside the Modal volume.
  - Utilities: `clts/utils/activations.py` provides `ActivationCacheConfig`, dataset/model helpers, and `compute_and_save_activations`.

- Training

  - `clts/utils/train.py`: Modal function `train` that loads cached activations , trains `CrossLayerTranscoder`, and logs metrics (Weights & Biases).

- Tests
  - `tests/`: Unit tests for encoders/decoders, loss, standardizers, and activation extraction.

## Resources

https://www.jonvet.com/blog/llm-transcoder-and-saes
https://transformer-circuits.pub/2024/crosscoders/index.html
