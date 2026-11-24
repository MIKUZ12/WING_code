# WING-LAM

WING-LAM learns camera and interaction latent factors from two RGB frames,
dense point tracks, and hand masks. The RGB-only student produces interaction
latents without trackers or masks and is required by the documented policy
target-extraction flow. Download placeholders are listed in the
[checkpoint guide](checkpoints.md).

## Python dependencies

The WING-LAM package and trainers require:

```text
numpy
torch
av
einops
```

Data preparation additionally requires `opencv-python-headless`. These packages
are included in the repository installation described in `docs/installation.md`.

The following model files and environments are external and are not distributed
with WING:

- A source LAM checkpoint compatible with `LAM_400k.ckpt`. The architecture
  needed to load it is included in `wing.latent_action.source_lam`.
- An AllTracker checkout and checkpoint for dense tracks.
- An EgoHOS checkout with its MMSegmentation environment, two-hand config, and
  checkpoint for hand masks.

Run AllTracker and EgoHOS extraction in their respective environments. The
remaining commands run in the WING environment from the repository root.

## Prepare supervision

Set portable paths in the shell:

```bash
VIDEO=path/to/video.mp4
CACHE_DIR=work/lam/example
ALLTRACKER_REPO=path/to/AllTracker
ALLTRACKER_CHECKPOINT=path/to/alltracker.pth
EGOHOS_REPO=path/to/EgoHOS/mmsegmentation
EGOHOS_CONFIG=path/to/seg_twohands_ccda.py
EGOHOS_CHECKPOINT=path/to/seg_twohands_ccda.pth
```

Extract aligned tracks and hand masks from the same frame range:

```bash
python scripts/lam/extract_alltracker_tracks.py \
  --input-video "$VIDEO" \
  --output "$CACHE_DIR/tracks.npz" \
  --alltracker-repo "$ALLTRACKER_REPO" \
  --checkpoint "$ALLTRACKER_CHECKPOINT" \
  --start-frame 0 \
  --num-frames 8 \
  --device cuda:0

python scripts/lam/extract_egohos_hand_masks.py \
  --input-video "$VIDEO" \
  --output "$CACHE_DIR/hand_masks.npz" \
  --egohos-repo "$EGOHOS_REPO" \
  --config "$EGOHOS_CONFIG" \
  --checkpoint "$EGOHOS_CHECKPOINT" \
  --start-frame 0 \
  --num-frames 8 \
  --device cuda:0
```

Build trajectory supervision:

```bash
python scripts/lam/build_wing_lam_pair_cache.py \
  --tracks "$CACHE_DIR/tracks.npz" \
  --hand-masks "$CACHE_DIR/hand_masks.npz" \
  --output-dir "$CACHE_DIR/pairs" \
  --gap 1 \
  --gap 4
```

Repeat these commands for each training clip. `--pair-cache` accepts either a
directory containing `pair_*.npz`, one pair archive, or a text file listing
pair archives relative to the list file.

## Train WING-LAM

```bash
LAM_CHECKPOINT=path/to/LAM_400k.ckpt

python scripts/lam/train_wing_lam.py \
  --pair-cache work/lam/train_pairs.txt \
  --validation-pair-cache work/lam/val_pairs.txt \
  --lam-checkpoint "$LAM_CHECKPOINT" \
  --output outputs/wing_lam.pt \
  --metrics-output outputs/wing_lam_metrics.jsonl \
  --steps 60000 \
  --batch-size 4 \
  --validate-every 1000 \
  --checkpoint-every 10000 \
  --device cuda:0
```

Use `--resume outputs/wing_lam_step00010000.pt` to resume the optimizer and
training step. Use `--init-wing-lam path/to/wing_lam.pt` to initialize only
the model weights for a new training stage.

## Train the RGB student

Export the frozen WING-LAM interaction targets for each split:

```bash
python scripts/lam/export_wing_lam_teacher_targets.py \
  --pair-list work/lam/train_pairs.txt \
  --wing-lam-checkpoint outputs/wing_lam.pt \
  --lam-checkpoint "$LAM_CHECKPOINT" \
  --split train \
  --output work/lam/train_targets.npz \
  --device cuda:0
```

Build the JSONL student index. Repeat `--source` for additional datasets or
splits:

```bash
python scripts/lam/build_wing_lam_student_index.py \
  --source "train::work/lam/train_pairs.txt::work/lam/train_targets.npz" \
  --output work/lam/student_index.jsonl
```

Train the two-frame RGB model:

```bash
python scripts/lam/train_wing_lam_rgb_student.py \
  --index work/lam/student_index.jsonl \
  --lam-checkpoint "$LAM_CHECKPOINT" \
  --init-wing-lam-checkpoint outputs/wing_lam.pt \
  --output outputs/wing_lam_rgb_student.pt \
  --steps 60000 \
  --batch-size 8 \
  --device cuda:0
```

For deployment or simulation target extraction, provide a JSONL manifest with
`video`, `frame_start`, `frame_end`, and `gap` fields:

```bash
python scripts/lam/label_videos_with_wing_lam_rgb_student.py \
  --manifest work/lam/deployment_pairs.jsonl \
  --checkpoint outputs/wing_lam_rgb_student.pt \
  --lam-checkpoint "$LAM_CHECKPOINT" \
  --output work/lam/deployment_latents.npz \
  --device cuda:0
```

The package entry points used by downstream code are
`wing.latent_action.load_rgb_student_checkpoint` and
`wing.latent_action.preprocess_rgb_frame`.

Teacher implementation: `wing.latent_action.lam`. Training dataset:
`wing.latent_action.lam_dataset`. Import `WINGLAM`, `WINGLAMLossWeights`,
`WINGLAMPairsDataset`, and `lam_loss` from `wing.latent_action`.

## Checkpoint compatibility

The package rename does not rewrite tensor keys. Source LAM checkpoints and
WING-LAM state dictionaries therefore load through the included architecture
without retaining the previous package name. Checkpoints store regular tensor
dictionaries rather than pickled model instances.

Student metadata uses `init_wing_lam_checkpoint`; exported teacher-target
metadata uses `wing_lam_checkpoint`. Re-export older assets with those field
names before using this release; no earlier teacher-field aliases are read.

Resuming restores model and optimizer state but starts a fresh deterministic
sampling stream; it is not bitwise replay of an uninterrupted run. Pair IDs in
newly generated student indices encode the source path and frame pair rather
than a content checksum. Keep indices with their corresponding target archives,
and rebuild the index when relocating or regenerating those training assets.

The source LAM architecture is adapted from DreamDojo components distributed
under Apache License 2.0; its license is retained at
`src/wing/latent_action/source_lam/LICENSE`.
