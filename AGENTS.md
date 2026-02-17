# Repository Guidelines

## Project Structure & Module Organization
- `nextgpt/` is the main Python package.
- `nextgpt/model/` contains model components; `nextgpt/dataset/` contains dataset loaders and samplers.
- Root entrypoints include `train_mem.py`, `train.py`, `predict.py`, `preprocess_embeddings.py`, and `merge_lora_weights.py`.
- Training workflows and Deepspeed configs live in `scripts/` (see `scripts/zero*.json`).
- Data and media assets are under `data/`, `assets/`, and `figures/`; legacy code is in `NExT-GPT-Lagacy/`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `python data/prepare_data.py`: preprocess downloaded datasets (see `data/DATA_README.md`).
- `python preprocess_embeddings.py <caption_json> <image|video|audio> <out_dir> <diffusion_model>`: precompute caption embeddings into `data/embed/`.
- `bash scripts/pretrain_enc.sh`: encoding-side alignment training.
- `bash scripts/pretrain_dec.sh`: decoding-side alignment training.
- `bash scripts/finetune.sh`: instruction tuning.
- `python predict.py`: run inference after checkpoints are placed under `./checkpoints/` and `./pretrain_ckpt/`.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and follows PEP 8 conventions.
- Use `snake_case` for functions/variables and `PascalCase` for classes.
- Keep dataset filenames aligned with existing patterns like `*_comprehension.json` and `*_generation.json`.
- No formatter or linter is enforced; keep diffs focused and avoid reformat-only changes.

## Testing Guidelines
- There is no automated test suite in this repo.
- Validate changes by running the relevant training or inference command and checking logs/outputs.

## Commit & Pull Request Guidelines
- Commit history favors short, imperative subjects without scopes (examples: `Update README.md`, `update data`).
- PRs should include a concise summary, impacted datasets/checkpoints, and the exact validation commands used.
- Do not commit large model weights or datasets; keep them in external storage and document paths.
