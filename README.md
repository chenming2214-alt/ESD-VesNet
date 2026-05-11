# ESD-VesNet

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
