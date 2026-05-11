# ESD-VesNet

<<<<<<< HEAD
**Paper title:** ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic Submucosal Dissection with Hard Negative Mining

ESD-VesNet is an uncertainty-aware vessel segmentation framework for endoscopic submucosal dissection (ESD). The model is trained with positive-negative learning and hard negative mining to suppress false positives from non-vessel structures while preserving vessel sensitivity.

## Contents

- `code/legacy_0105/train_vessel_esd_edl_hnm_fpaware_fullsam_0105.py`: main ESD-VesNet training entrypoint.
- `code/tools/eval_val_metrics.py`: validation/evaluation script for Dice, IoU, VDR, S-measure, E-measure, and MAE.
- `code/tools/infer_overlay_smooth.py`: image/video-style vessel mask inference visualization.
- `code/tools/infer_centerline_visualize.py`: image centerline visualization from vessel predictions.
- `code/tools/infer_centerline_video.py`: real-time style video centerline inference.
- `code/save/vessel_esd_sam3_fullsam_edl_hnm_0105/best_fullsam_0105.pth`: trained ESD-VesNet checkpoint.

## Model

The submitted checkpoint corresponds to the ESD-VesNet model used in the paper:

`ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic Submucosal Dissection with Hard Negative Mining`

The implementation includes evidential uncertainty estimation, uncertainty-gated prediction, positive-negative learning, and hard negative mining for ESD vessel segmentation.

## Notes

Dataset files and large intermediate outputs are intentionally excluded from this submission package. The `sam3-main` directory is kept as an internal dependency directory for model loading compatibility.
=======
Official implementation of **ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic Submucosal Dissection with Hard Negative Mining**, accepted by **International Journal of Computer Assisted Radiology and Surgery (IJCARS)**.

ESD-VesNet is designed for vessel segmentation in endoscopic submucosal dissection (ESD). The method combines SAM3-based visual representations, evidential uncertainty estimation, uncertainty-gated prediction, positive-negative learning, and hard negative mining to suppress false positives from non-vessel structures while preserving vessel sensitivity.

## News

- The code and trained checkpoint are released for academic research and reproducibility.
- The full dataset is being prepared for the journal article and is expected to be released to the community within one year.
- The test set used for paper-level reproducibility is available upon reasonable request by email to the authors.

## Repository Structure

```text
.
├── legacy_0105/
│   └── train_vessel_esd_edl_hnm_fpaware_fullsam_0105.py
├── models/
├── scripts/
│   └── train_edl_hnm_fullsam_0105.sh
├── tools/
│   ├── eval_val_metrics.py
│   ├── infer_overlay_smooth.py
│   ├── infer_centerline_visualize.py
│   └── infer_centerline_video.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/chenming2214-alt/ESD-VesNet.git
cd ESD-VesNet

conda create -n esd-vesnet python=3.9 -y
conda activate esd-vesnet
```

Install PyTorch according to your CUDA version. For the environment used in this release:

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

If you use a different CUDA version, please install the matching PyTorch build from the [official PyTorch installation page](https://pytorch.org/get-started/locally/) first, then install the remaining dependencies from `requirements.txt`.

## Model Weights

Two sets of weights are needed for reproduction.

1. **ESD-VesNet checkpoint**

   Download the trained ESD-VesNet checkpoint from Hugging Face:

   [best_fullsam_0105.pth](https://huggingface.co/chenhuansheng/ESD-VesNet/blob/main/best_fullsam_0105.pth)

   Place it under:

   ```text
   save/vessel_esd_sam3_fullsam_edl_hnm_0105/best_fullsam_0105.pth
   ```

2. **Official SAM3 checkpoint**

   Download the official SAM3 checkpoint from Meta's Hugging Face repository:

   [facebook/sam3](https://huggingface.co/facebook/sam3)

   The SAM3 repository is gated on Hugging Face, so you may need to log in and request/accept access before downloading. Place the downloaded checkpoint as:

   ```text
   checkpoints/sam3.pt
   ```

## Data Availability

The ESD vessel dataset is currently being organized for the journal article. We plan to release the dataset to the research community within one year.

For reproducibility of the reported test-set results, please contact the authors by email to request access to the test set. Access may be subject to institutional, ethical, and data-sharing requirements.

After obtaining the data, organize it according to the path configuration used in `legacy_0105/train_vessel_esd_edl_hnm_fpaware_fullsam_0105.py`, or update the `DATA_ROOTS` and negative-data paths in the same file to match your local dataset location.

## Quick Start

### 1. Test-Set Inference / Evaluation

```bash
cd /path/to/ESD-VesNet
python tools/eval_val_metrics.py \
  --ckpt save/vessel_esd_sam3_fullsam_edl_hnm_0105/best_fullsam_0105.pth \
  --model sam3-sam-edl --inp-size 1024 --device cuda \
  --tta-mode hv --use-gated --post-close-k 7 --thr 0.43
```

If you keep the checkpoint in another location, replace the value of `--ckpt` with the absolute path to `best_fullsam_0105.pth`.

Optional: save per-image metrics to a CSV file.

```bash
python tools/eval_val_metrics.py \
  --ckpt save/vessel_esd_sam3_fullsam_edl_hnm_0105/best_fullsam_0105.pth \
  --model sam3-sam-edl --inp-size 1024 --device cuda \
  --tta-mode hv --use-gated --post-close-k 7 --thr 0.43 \
  --out-csv save/eval_results.csv
```

### 2. Run Inference on Images

```bash
python tools/infer_overlay_smooth.py \
  --image path/to/image.png \
  --ckpt save/vessel_esd_sam3_fullsam_edl_hnm_0105/best_fullsam_0105.pth \
  --model sam3-sam-edl \
  --use-gated \
  --out-dir save/overlay_smooth
```

For a folder of images:

```bash
python tools/infer_overlay_smooth.py \
  --data-dir path/to/images \
  --ckpt save/vessel_esd_sam3_fullsam_edl_hnm_0105/best_fullsam_0105.pth \
  --model sam3-sam-edl \
  --use-gated \
  --out-dir save/overlay_smooth
```

### 3. Train / Fine-Tune

Before training, update the dataset paths in `legacy_0105/train_vessel_esd_edl_hnm_fpaware_fullsam_0105.py`.

Single-GPU training:

```bash
bash scripts/train_edl_hnm_fullsam_0105.sh \
  --gpus 1 \
  --devices 0 \
  --batch-size 4
```

Multi-GPU training:

```bash
bash scripts/train_edl_hnm_fullsam_0105.sh \
  --gpus 3 \
  --devices 0,1,2 \
  --hnm-scan 600 \
  --batch-size 4
```

Resume or evaluate from a checkpoint:

```bash
bash scripts/train_edl_hnm_fullsam_0105.sh \
  --gpus 1 \
  --devices 0 \
  --eval-only \
  --ckpt save/vessel_esd_sam3_fullsam_edl_hnm_0105/best_fullsam_0105.pth
```

## Reproducibility Notes

- The released ESD-VesNet checkpoint is the model used for the paper experiments.
- Large checkpoints, intermediate outputs, and dataset files are intentionally excluded from the GitHub repository.
- For exact paper-level test-set reproduction, please request the test set from the authors.
- The SAM3 base checkpoint must be downloaded separately from the official `facebook/sam3` Hugging Face repository.

## Citation

If you find this repository useful, please cite our paper. The BibTeX entry will be updated after the final article metadata is available.

```bibtex
@article{esdvesnet2026,
  title={ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic Submucosal Dissection with Hard Negative Mining},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2026}
}
```

## License

This repository is released for academic research use. Please check the license file and the terms of the official SAM3 model before use.
>>>>>>> 4f0d0e6 (Update README for open-source release)
