# Quickstart: Pix2Pix (2D and 3D) on your own medical data

This is a short walkthrough for training Pix2Pix in this repo on a paired MR
image-translation task (e.g. axial Dixon → T2 sagittal, or MR → CT). It uses
the same code paths as the STIR / T1w / T2w models from the paper *"Generating
synthetic high-resolution spinal STIR and T1w images from T2w FSE and
low-resolution axial Dixon"*.

**Recommendation:** always start with the **2D** config. It converges faster,
uses far less GPU memory, and lets you verify that your data pairing and
preprocessing are correct before moving to the 3D variant.

---

## 1. Install

```bash
conda env create -f environment_small.yml
conda activate diffusion3D
```

## 2. Prepare your data — you have to write two small scripts

The dataloader does **not** read raw NIfTI files directly. You need two things:

1. **Preprocessed slices/volumes as `.npz` files**, one per sample, with keys
   matching your `condition_types` in the config (first key = target modality,
   rest = input channels).
2. **An `.xlsx` file** listing the paths to those `.npz` files together with a
   `Phase` column (`train` / `val` / `test`), which the dataloader consumes.

There is **no automatic conversion** — you have to write both scripts once for
your dataset. Use the existing scripts as templates:

- **Slice / volume generation** → look at
  [`scripts/datagen/`](scripts/datagen/), e.g.
  [`dataset_gen_spinegan_T2w.py`](scripts/datagen/dataset_gen_spinegan_T2w.py)
  (2D) and [`dataset_generation3D.py`](scripts/datagen/dataset_generation3D.py)
  (3D). These scripts read the raw `.nii.gz` files, do axis alignment,
  normalization and cropping, and save one `.npz` per slice/volume containing
  all requested modalities under their key names.
- **CSV / XLSX index** → look at [`scripts/csv_gen/`](scripts/csv_gen/), e.g.
  [`make_csv_stir.py`](scripts/csv_gen/make_csv_stir.py). These scripts walk
  through your BIDS-style dataset, split into train/val/test, resample the
  NIfTIs to a common voxel grid (so target and inputs share the same shape)
  and write the resulting `.xlsx` file that the config's `dataset:` field
  points to.

Registration between the input modality (e.g. axial Dixon) and the target
modality (e.g. sagittal T2) must be done up-front — Pix2Pix is a *paired*
method, so voxels at the same position in both volumes must correspond.

## 3. Adjust the config

Two example configs are provided for the task *axial Dixon → T2 sagittal*.
Copy them and edit `dataset:`, `condition_types:`, `gpus:` and, for 3D,
`target_shape3D:` / `batch_size:` to fit your data and GPU:

- **2D (start here):**
  [`config/2D/pix2pix_dixon_to_t2sag_example.conf`](config/2D/pix2pix_dixon_to_t2sag_example.conf)
- **3D (only after 2D works):**
  [`config/3D/pix2pix_dixon_to_t2sag_example.conf`](config/3D/pix2pix_dixon_to_t2sag_example.conf)

The first entry in `condition_types` is the target modality; the following
entries are the input channels concatenated to the generator input. The key
names must match the keys in your `.npz` files.

## 4. Train

```bash
python CUT.py --config config/2D/pix2pix_dixon_to_t2sag_example.conf --new --gpus 0
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs_diffusion3D --port 6007
```

## 5. Move to 3D once 2D is good

Once the 2D model produces reasonable outputs, switch to the 3D config. Adjust:

- `target_shape3D: [D, H, W]` — start small (e.g. `[16, 256, 256]`) and only
  increase `D` if VRAM allows.
- `batch_size` — often needs to drop to `1` for 3D.
- Everything else can stay the same as in the 2D config.

## Notes on modes

- `mode: pix2pix` — paired image-to-image translation (this quickstart).
- `mode: cut` — unpaired translation (Contrastive Unpaired Translation).
- `mode: paired_cut` — CUT with the paired L1/SSIM losses added.

Start file for all three is `CUT.py` (the name is historical — Pix2Pix runs
from the same entry point, controlled by `mode:` in the config).
