# Running the GPU jobs on Ada

Ada handles every job that needs a large GPU or lots of disk: the reproducibility check of the dataset,
feature extraction for the three models, the layer sweep, and all representation-probe (RP) runs.
Transcription and the transcript probes run on Mohit's PC. Results meet in this repo under `results/`.

Everything below is one command per step. Each step says what it produces and how to check it.

## 0. Before starting

- Hugging Face: log in, open https://huggingface.co/datasets/ARTPARK-IISc/Vaani and accept the terms
  (approved automatically). Create a read token at https://huggingface.co/settings/tokens.
  Also request https://huggingface.co/datasets/changelinglab/vaani-hi-geo (manual approval; not needed to start).
- Storage: about **170 GB** in a directory that persists between jobs (not node-local `/scratch`):
  12 GB raw Vaani shards, 4.6 GB dataset, and roughly 15 GB per cached layer per model.

## 1. Code and environment (once, about 15 min)

```bash
cd <your 170 GB directory>
git clone https://github.com/unignoramus11/sal-project.git && cd sal-project
bash tools/setup_env.sh              # needs internet; run on the login node or an interactive job
envs/prism/bin/hf auth login         # paste the Hugging Face token
cp slurm/ada.env.example slurm/ada.env && nano slurm/ada.env   # account, partition, GPU type, WORK dir
```

`setup_env.sh` ends with `environment OK` and prints the torch, k2 and transformers versions. On the login
node `cuda False` is expected; on a GPU node it must say `cuda True`.

## 2. Rebuild the dataset and check it (CPU job, about 1-2 h)

```bash
GPU=0 bash slurm/submit.sh slurm/rebuild.sbatch
tail -4 outputs/slurm/rebuild_<jobid>.out
```

The two `dataset`/`expected` lines and the two `splits`/`expected` lines must be identical. If they are,
the dataset is bit-for-bit the one used on the PC, which is our reproducibility check. If they differ, stop
and send the whole log. (If compute nodes have no internet, run the two Python lines of `rebuild.sbatch`
on the login node instead.)

## 3. Extract features (3 GPU jobs, about 1-2 h each)

```bash
for M in w2v2 whisper zipa; do bash slurm/submit.sh slurm/extract.sbatch $M; done
```

Each writes `outputs/features/<model>_rebuild/`: `pooled.npy` (every layer, time-averaged), `frames_L<l>.npy`
(the last layer's frames), `index.parquet`, `meta.json`. The log ends with a JSON summary; `n_utterances`
must be 22551.

## 4. Layer sweep (3 GPU jobs, under 1 h each)

```bash
for M in w2v2 whisper zipa; do bash slurm/submit.sh slurm/sweep.sbatch $M; done
```

Writes `results/layer_sweep/<model>_rebuild_{speaker_split,prism}.csv` and prints the top layers. Push these
(step 7); the top 2-3 layers per model decide which extra frame layers to cache next.

## 5. RP baselines for the mid evaluation (6 GPU jobs)

```bash
bash slurm/rp_baselines.sh
```

PRiSM's RP-Attn on the last layer: 3 models x 2 split protocols x 5 seeds. Each run writes
`results/runs/rp_<model>_rebuild_L<l>_attn_<protocol>_s<seed>/` with `metrics.json`, `history.csv` and
`predictions.parquet`.

## 6. Main RP sweep (after the layer sweep)

Cache the chosen layers' frames (`bash slurm/submit.sh slurm/extract.sbatch w2v2 8 16 24`, which re-extracts
and writes those layers), then submit the probe x layer-setting x protocol grid with `slurm/rp.sbatch`. The
exact list will be in `slurm/rp_sweep.sh` once the layers are chosen.

## 7. Sending results back

```bash
git pull --rebase
git add results/
git commit -m "Add <what> results"
git push
```

Only `results/` is versioned. Features and data stay on Ada (they are gitignored and too large).

## 8. Please report once

So the job scripts fit the cluster:
1. `sinfo -o "%P %G %l"`: partitions, GPU types, time limits.
2. `nvidia-smi` on a GPU node: driver version (needs to support CUDA 12.6, i.e. driver 560 or newer).
3. Whether compute nodes can reach huggingface.co (`curl -sI https://huggingface.co | head -1` inside a job).
4. Disk quota of the directory you use.

## Troubleshooting

- `CUDA error: no kernel image is available`: the GPU is too old for the torch build; report the GPU model.
- `Access denied` from Hugging Face: the Vaani terms were not accepted for the logged-in account.
- Out of GPU memory in an RP job: add `--batch-size 24` at the end of the `python` line in `slurm/rp.sbatch`.
