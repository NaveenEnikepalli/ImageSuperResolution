"""Comprehensive Super-Resolution Model Evaluation Pipeline.

Author: Antigravity
Purpose: Fair, rigorous evaluation comparing:
  - Student x2 vs Pretrained SwinIR x2 on Set5 & Set14
  - Student x4 vs Pretrained SwinIR x4 on Set5 & Set14
Computes: PSNR, SSIM, MAE, RMSE, inference latency, parameter counts, and checkpoint sizes.
Saves raw per-image metrics to JSON and outputs formatted benchmark summaries.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import piq

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.models.student_model import StudentModel
from training.teacher.swinir_wrapper import SwinIRWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EvaluateSR")


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute PSNR, SSIM, MAE, and RMSE between prediction and target tensors.

    Args:
        pred: Super-resolved tensor (1, 3, H, W) in [0.0, 1.0].
        target: Ground truth HR tensor (1, 3, H, W) in [0.0, 1.0].

    Returns:
        Dict with psnr, ssim, mae, rmse.
    """
    pred_c = torch.clamp(pred, 0.0, 1.0)
    target_c = torch.clamp(target, 0.0, 1.0)

    psnr_val = float(piq.psnr(pred_c, target_c, data_range=1.0).item())
    ssim_val = float(piq.ssim(pred_c, target_c, data_range=1.0).item())
    mae_val = float(torch.mean(torch.abs(pred_c - target_c)).item())
    rmse_val = float(torch.sqrt(torch.mean((pred_c - target_c) ** 2)).item())

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mae": mae_val,
        "rmse": rmse_val,
        "lpips": "N/A (library unavailable)"
    }


def load_student_model(scale: int, checkpoint_path: Path, device: torch.device) -> StudentModel:
    """Instantiate and load weights for Student Model."""
    logger.info(f"Loading Student x{scale} from {checkpoint_path}...")
    model = StudentModel(
        in_channels=3,
        out_channels=3,
        num_features=48,
        distilled_channels=24,
        num_blocks=3,
        scale=scale,
        negative_slope=0.05
    )
    if checkpoint_path.exists():
        raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = raw
        if isinstance(raw, dict):
            for k in ["student_state_dict", "state_dict", "model_state_dict", "model"]:
                if k in raw and isinstance(raw[k], dict):
                    state_dict = raw[k]
                    break
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
    else:
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model


def load_swinir_model(scale: int, checkpoint_path: Path, device: torch.device) -> SwinIRWrapper:
    """Instantiate and load weights for SwinIR Model via SwinIRWrapper."""
    logger.info(f"Loading SwinIR x{scale} from {checkpoint_path}...")
    teacher = SwinIRWrapper(scale=scale, in_channels=3)
    if checkpoint_path.exists():
        teacher.load_checkpoint(checkpoint_path)
    else:
        raise FileNotFoundError(f"SwinIR checkpoint not found: {checkpoint_path}")

    teacher.to(device)
    teacher.eval()
    return teacher


def evaluate_dataset_pair(
    dataset_name: str,
    dataset_dir: Path,
    scale: int,
    student_model: StudentModel,
    swinir_model: SwinIRWrapper,
    device: torch.device,
    save_samples_dir: Path = None,
) -> Dict[str, Any]:
    """Evaluate Student and SwinIR on a single benchmark dataset under identical conditions.

    Args:
        dataset_name: Name of dataset ('Set5' or 'Set14').
        dataset_dir: Path to directory containing HR images.
        scale: Upscaling factor (2 or 4).
        student_model: Evaluated student model instance.
        swinir_model: Evaluated SwinIR reference model instance.
        device: Torch compute device.
        save_samples_dir: Directory to save representative output comparisons.

    Returns:
        Dictionary containing aggregate summary and per-image details.
    """
    image_paths = sorted(list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.bmp")))
    if not image_paths:
        raise ValueError(f"No benchmark images found in {dataset_dir}")

    per_image_results = []
    student_times = []
    swinir_times = []

    student_metrics_list = []
    swinir_metrics_list = []

    if save_samples_dir:
        save_samples_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        filename = img_path.name
        img_hr_pil = Image.open(img_path).convert("RGB")
        w, h = img_hr_pil.size

        # Modcrop to ensure spatial dimensions are strictly divisible by scale
        w_mod = (w // scale) * scale
        h_mod = (h // scale) * scale
        if (w_mod, h_mod) != (w, h):
            img_hr_pil = img_hr_pil.crop((0, 0, w_mod, h_mod))

        # Standard bicubic downsampling (pure benchmark degradation)
        lr_w, lr_h = w_mod // scale, h_mod // scale
        img_lr_pil = img_hr_pil.resize((lr_w, lr_h), Image.BICUBIC)

        # Convert to tensor (1, 3, H, W) normalized to [0.0, 1.0]
        hr_tensor = torch.from_numpy(np.array(img_hr_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        lr_tensor = torch.from_numpy(np.array(img_lr_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        lr_tensor = lr_tensor.to(device)
        hr_tensor = hr_tensor.to(device)

        # 1. Student Inference
        with torch.no_grad():
            t0 = time.perf_counter()
            sr_student = student_model(lr_tensor)
            sr_student = torch.clamp(sr_student, 0.0, 1.0)
            t_student = time.perf_counter() - t0

        student_times.append(t_student)
        m_student = compute_metrics(sr_student, hr_tensor)
        student_metrics_list.append(m_student)

        # 2. SwinIR Inference
        with torch.no_grad():
            t0 = time.perf_counter()
            sr_swinir = swinir_model(lr_tensor)
            sr_swinir = torch.clamp(sr_swinir, 0.0, 1.0)
            t_swinir = time.perf_counter() - t0

        swinir_times.append(t_swinir)
        m_swinir = compute_metrics(sr_swinir, hr_tensor)
        swinir_metrics_list.append(m_swinir)

        # Verify exact output dimensions
        assert sr_student.shape == (1, 3, h_mod, w_mod), f"Student output shape {sr_student.shape} mismatch"
        assert sr_swinir.shape == (1, 3, h_mod, w_mod), f"SwinIR output shape {sr_swinir.shape} mismatch"

        # Save representative sample images (e.g. for first 2 images)
        if save_samples_dir and idx < 2:
            base_name = img_path.stem
            s_pil = Image.fromarray((sr_student.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8))
            sw_pil = Image.fromarray((sr_swinir.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8))
            s_pil.save(save_samples_dir / f"{base_name}_x{scale}_student.png")
            sw_pil.save(save_samples_dir / f"{base_name}_x{scale}_swinir.png")

        per_image_results.append({
            "image": filename,
            "input_resolution": f"{lr_w}x{lr_h}",
            "output_resolution": f"{w_mod}x{h_mod}",
            "student": {
                **m_student,
                "inference_time_sec": t_student
            },
            "swinir": {
                **m_swinir,
                "inference_time_sec": t_swinir
            }
        })

        logger.info(
            f"[{dataset_name} x{scale}] {filename:<12} | "
            f"Student PSNR: {m_student['psnr']:.2f} dB, SSIM: {m_student['ssim']:.4f}, Time: {t_student:.3f}s | "
            f"SwinIR PSNR: {m_swinir['psnr']:.2f} dB, SSIM: {m_swinir['ssim']:.4f}, Time: {t_swinir:.3f}s"
        )

    # Compute dataset aggregated averages
    student_summary = {
        "psnr": float(np.mean([m["psnr"] for m in student_metrics_list])),
        "ssim": float(np.mean([m["ssim"] for m in student_metrics_list])),
        "mae": float(np.mean([m["mae"] for m in student_metrics_list])),
        "rmse": float(np.mean([m["rmse"] for m in student_metrics_list])),
        "lpips": "N/A (library unavailable)",
        "avg_inference_time_sec": float(np.mean(student_times))
    }

    swinir_summary = {
        "psnr": float(np.mean([m["psnr"] for m in swinir_metrics_list])),
        "ssim": float(np.mean([m["ssim"] for m in swinir_metrics_list])),
        "mae": float(np.mean([m["mae"] for m in swinir_metrics_list])),
        "rmse": float(np.mean([m["rmse"] for m in swinir_metrics_list])),
        "lpips": "N/A (library unavailable)",
        "avg_inference_time_sec": float(np.mean(swinir_times))
    }

    return {
        "dataset": dataset_name,
        "scale": scale,
        "num_images": len(image_paths),
        "student_summary": student_summary,
        "swinir_summary": swinir_summary,
        "per_image_results": per_image_results
    }


def run_full_evaluation() -> Dict[str, Any]:
    """Execute evaluation matrix covering all 4 target configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Super-Resolution Benchmark Evaluation on device: {device}")

    # Paths
    checkpoints_dir = ROOT_DIR / "checkpoints"
    student_x2_ckpt = checkpoints_dir / "student" / "best_student_x2.pth"
    student_x4_ckpt = checkpoints_dir / "student" / "best_student_x4.pth"
    swinir_x2_ckpt = checkpoints_dir / "teacher" / "swinir_x2.pth"
    swinir_x4_ckpt = checkpoints_dir / "teacher" / "swinir_x4.pth"

    set5_dir = ROOT_DIR / "datasets" / "Set5"
    set14_dir = ROOT_DIR / "datasets" / "Set14"

    eval_out_dir = ROOT_DIR / "details" / "evaluation"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = eval_out_dir / "samples"

    # Model complexity metrics
    def model_stats(model: torch.nn.Module, ckpt_path: Path):
        tot_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else 0.0
        return {
            "total_parameters": tot_params,
            "trainable_parameters": train_params,
            "checkpoint_size_mb": ckpt_size_mb
        }

    # 1. Load Scale 2 Models
    logger.info("\n=======================================================")
    logger.info("CONFIGURATION A: SCALE x2 (Student x2 vs SwinIR x2)")
    logger.info("=======================================================")
    student_x2 = load_student_model(scale=2, checkpoint_path=student_x2_ckpt, device=device)
    swinir_x2 = load_swinir_model(scale=2, checkpoint_path=swinir_x2_ckpt, device=device)

    complexity_student_x2 = model_stats(student_x2, student_x2_ckpt)
    complexity_swinir_x2 = model_stats(swinir_x2, swinir_x2_ckpt)

    res_x2_set5 = evaluate_dataset_pair("Set5", set5_dir, 2, student_x2, swinir_x2, device, samples_dir)
    res_x2_set14 = evaluate_dataset_pair("Set14", set14_dir, 2, student_x2, swinir_x2, device, samples_dir)

    # 2. Load Scale 4 Models
    logger.info("\n=======================================================")
    logger.info("CONFIGURATION B: SCALE x4 (Student x4 vs SwinIR x4)")
    logger.info("=======================================================")
    student_x4 = load_student_model(scale=4, checkpoint_path=student_x4_ckpt, device=device)
    swinir_x4 = load_swinir_model(scale=4, checkpoint_path=swinir_x4_ckpt, device=device)

    complexity_student_x4 = model_stats(student_x4, student_x4_ckpt)
    complexity_swinir_x4 = model_stats(swinir_x4, swinir_x4_ckpt)

    res_x4_set5 = evaluate_dataset_pair("Set5", set5_dir, 4, student_x4, swinir_x4, device, samples_dir)
    res_x4_set14 = evaluate_dataset_pair("Set14", set14_dir, 4, student_x4, swinir_x4, device, samples_dir)

    # Complete results payload
    evaluation_payload = {
        "meta": {
            "device": str(device),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "model_complexity": {
            "student_x2": complexity_student_x2,
            "swinir_x2": complexity_swinir_x2,
            "student_x4": complexity_student_x4,
            "swinir_x4": complexity_swinir_x4
        },
        "results": {
            "scale_2": {
                "Set5": res_x2_set5,
                "Set14": res_x2_set14
            },
            "scale_4": {
                "Set5": res_x4_set5,
                "Set14": res_x4_set14
            }
        }
    }

    # Save raw json output
    json_path = eval_out_dir / "raw_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)
    logger.info(f"\nRaw per-image results successfully saved to: {json_path}")

    return evaluation_payload


if __name__ == "__main__":
    run_full_evaluation()
