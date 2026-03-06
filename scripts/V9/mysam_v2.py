# segment_and_evaluate_msd_3d_v2.py

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"导入SAM2库时出错。请确保'sam2-image-segmentation'已通过pip正确安装。")
    raise e

DATA_ROOT = Path(r"D:\Datasets\MSD")
PROMPT_ROOT = Path(r"D:\Datasets\prompt\yoloworld\YoloWorld\MSD_results")
RESULTS_ROOT = Path(r"D:\Datasets\results\YOLO-World\MSD")

SAM2_MODEL_CFG = Path(r"D:\Codes\PaperCodes\AAAI_v2\sam2\config.yaml")
SAM2_CHECKPOINT = Path(r"D:\Codes\PaperCodes\AAAI_v2\sam2\sam2.1_hiera_base_plus.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASKS = ["Task02_Heart", "Task04_Hippocampus", "Task05_Prostate", "Task09_Spleen"]



def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    epsilon = 1e-8
    return intersection / (union + epsilon)


def calculate_3d_metrics(pred_volume: np.ndarray, gt_volume: np.ndarray) -> Optional[Dict[str, float]]:
    if pred_volume.shape != gt_volume.shape:
        print(f"错误：预测体和标签体的尺寸不匹配: {pred_volume.shape} vs {gt_volume.shape}")
        return None

    pred_mask = pred_volume.astype(bool)
    gt_mask = gt_volume.astype(bool)
    epsilon = 1e-8

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dsc = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + epsilon)

    if not np.any(gt_mask):
        return {"dsc": dsc, "nsd": 1.0 if not np.any(pred_mask) else 0.0}

    dist_gt = distance_transform_edt(np.logical_not(gt_mask))
    dist_pred = distance_transform_edt(np.logical_not(pred_mask))

    surface_gt = (dist_gt == 1)
    surface_pred = (dist_pred == 1)

    if not np.any(surface_pred):
        return {"dsc": dsc, "nsd": 0.0}

    if not np.any(surface_gt) or not np.any(surface_pred):
        return {"dsc": dsc, "nsd": 1.0 if np.array_equal(pred_mask, gt_mask) else 0.0}

    dist_pred_to_gt = dist_pred[surface_gt]
    dist_gt_to_pred = dist_gt[surface_pred]

    tau = 1.0
    nsd_gt_to_pred = np.sum(dist_gt_to_pred <= tau) / len(dist_gt_to_pred)
    nsd_pred_to_gt = np.sum(dist_pred_to_gt <= tau) / len(dist_pred_to_gt)

    nsd = 0.5 * (nsd_gt_to_pred + nsd_pred_to_gt)
    return {"dsc": dsc, "nsd": nsd}


def run_3d_evaluation(pred_task_dir: Path, gt_task_dir: Path) -> Optional[Dict[str, float]]:
    print(f"\n--- 开始对任务 '{pred_task_dir.name}' 进行3D评测 ---")

    synced_output_dir = pred_task_dir / "synced_for_eval"
    synced_output_dir.mkdir(exist_ok=True, parents=True)

    gt_case_folders = sorted([d for d in (gt_task_dir / "labelsTr").iterdir() if d.is_dir()])
    if not gt_case_folders:
        print(f"错误：在 '{gt_task_dir}' 中未找到任何病例（子文件夹）。")
        return None

    all_case_metrics = {"dsc": [], "nsd": []}

    for gt_case_dir in tqdm(gt_case_folders, desc=f"评测病例 ({pred_task_dir.name})"):
        case_name = gt_case_dir.name
        pred_case_dir = pred_task_dir / case_name
        output_case_dir = synced_output_dir / case_name
        output_case_dir.mkdir(exist_ok=True)

        if not pred_case_dir.is_dir():
            print(f"警告：找不到对应的预测病例文件夹: '{pred_case_dir}'，已跳过。")
            continue

        gt_slices = sorted([f for f in gt_case_dir.iterdir() if f.suffix.lower() == '.jpg'])
        synced_pred_slices_for_stacking, gt_slices_for_stacking = [], []

        for gt_slice_path in gt_slices:
            pred_slice_path = pred_case_dir / gt_slice_path.name
            output_slice_path = output_case_dir / gt_slice_path.name

            if not pred_slice_path.exists():
                print(f"警告：找不到对应的预测切片: '{pred_slice_path}'，跳过。")
                continue

            try:
                with Image.open(gt_slice_path) as gt_img:
                    gt_img_l = gt_img.convert('L')
                    gt_mask_np = np.array(gt_img_l) > 50

                    if not np.any(gt_mask_np):
                        black_img = Image.new('L', gt_img_l.size, 0)
                        black_img.save(output_slice_path, "JPEG")
                        synced_pred_slices_for_stacking.append(np.zeros_like(gt_mask_np))
                    else:
                        shutil.copy(pred_slice_path, output_slice_path)
                        pred_img = Image.open(pred_slice_path).convert('L')
                        synced_pred_slices_for_stacking.append(np.array(pred_img) > 50)

                    gt_slices_for_stacking.append(gt_mask_np)
            except Exception as e:
                print(f"错误：处理切片 '{gt_slice_path}' 时发生错误: {e}")
                continue

        if not synced_pred_slices_for_stacking or not gt_slices_for_stacking:
            continue

        try:
            pred_volume = np.stack(synced_pred_slices_for_stacking, axis=-1)
            gt_volume = np.stack(gt_slices_for_stacking, axis=-1)

            case_metrics = calculate_3d_metrics(pred_volume, gt_volume)
            if case_metrics:
                all_case_metrics["dsc"].append(case_metrics["dsc"])
                all_case_metrics["nsd"].append(case_metrics["nsd"])
        except Exception as e:
            print(f"错误：为病例 {case_name} 计算3D指标时出错: {e}")

    if not all_case_metrics["dsc"]:
        print("错误：未能成功评测任何病例。")
        return None

    return {
        "avg_dsc": np.mean(all_case_metrics["dsc"]),
        "avg_nsd": np.mean(all_case_metrics["nsd"]),
        "case_count": len(all_case_metrics["dsc"])
    }


def load_prompts(prompt_file: Path) -> Optional[np.ndarray]:
    if not prompt_file.exists(): return None
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if content.lower() == "null": return None
    boxes = []
    for line in content.split('\n'):
        if not line.strip(): continue
        try:
            coords = [float(c) for c in line.strip().split()]
            if len(coords) == 4: boxes.append(coords)
        except ValueError:
            continue
    return np.array(boxes) if boxes else None


def main():
    print(f"使用设备: {DEVICE}")

    print("正在加载SAM2模型...")
    try:
        sam2_model = build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT))
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        sam2_predictor.model.to(DEVICE)
        print("✅ SAM2模型加载成功。")
    except Exception as e:
        print(f"❌ 加载SAM2模型失败。错误: {e}")
        return

    for task_name in TASKS:
        print(f"\n{'=' * 20} 开始处理任务: {task_name} {'=' * 20}")

        image_dir = DATA_ROOT / task_name / "imagesTr"
        prompt_dir = PROMPT_ROOT / task_name / "imagesTr"
        gt_dir = DATA_ROOT / task_name / "labelsTr"
        output_dir = RESULTS_ROOT / task_name
        output_dir.mkdir(parents=True, exist_ok=True)

        case_folders = sorted([d for d in image_dir.iterdir() if d.is_dir()])
        if not case_folders:
            print(f"警告：在 {image_dir} 中未找到任何病例文件夹，跳过此任务。")
            continue

        segmentation_start_time = time.perf_counter()

        for case_dir in tqdm(case_folders, desc=f"分割病例 ({task_name})"):
            case_name = case_dir.name
            (output_dir / case_name).mkdir(exist_ok=True)

            image_files = sorted([f for f in case_dir.iterdir() if f.suffix.lower() == '.jpg'])

            for image_path in image_files:
                try:
                    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                        image = Image.open(image_path).convert("RGB")
                        sam2_predictor.set_image(image)

                        prompt_file = prompt_dir / case_name / f"{image_path.stem}.txt"
                        box_prompts = load_prompts(prompt_file)

                        best_mask = None
                        if box_prompts is not None:
                            masks, scores, _ = sam2_predictor.predict(box=box_prompts, multimask_output=False)
                            best_mask = masks[np.argmax(scores)]
                        else:
                            gt_path = gt_dir / case_name / image_path.name
                            if not gt_path.exists(): continue

                            gt_image = Image.open(gt_path).convert('L')
                            gt_mask = np.array(gt_image) > 50

                            masks, _, _ = sam2_predictor.predict(multimask_output=True)

                            max_iou = -1.0
                            for candidate_mask in masks:
                                iou = calculate_iou(candidate_mask, gt_mask)
                                if iou > max_iou:
                                    max_iou = iou
                                    best_mask = candidate_mask

                        if best_mask is not None:
                            # 【核心修复1】使用 np.squeeze() 移除多余维度
                            squeezed_mask = np.squeeze(best_mask)
                            mask_img = Image.fromarray(squeezed_mask.astype(np.uint8) * 255)

                            output_path = output_dir / case_name / f"{image_path.stem}.jpg"
                            mask_img.save(output_path, "JPEG")

                except Exception as e:
                    print(f"错误：处理文件 {image_path} 时出错: {e}")

        segmentation_end_time = time.perf_counter()
        total_seg_time = segmentation_end_time - segmentation_start_time
        print(f"✅ 任务 '{task_name}' 分割完成，总耗时 {total_seg_time:.2f} 秒。")

        final_metrics = run_3d_evaluation(output_dir, DATA_ROOT / task_name)

        report_content = (
            f"任务 '{task_name}' 的评测报告\n"
            f"=========================================\n"
            f"总分割耗时: {total_seg_time:.2f} 秒\n"
        )
        if final_metrics:
            report_content += (
                f"处理的病例总数: {final_metrics['case_count']}\n"
                f"-----------------------------------------\n"
                f"平均 3D Dice Score (DSC): {final_metrics['avg_dsc']:.4f}\n"
                f"平均 3D Normalized Surface Distance (NSD): {final_metrics['avg_nsd']:.4f}\n"
            )
        else:
            report_content += "评测失败，未能生成任何指标。\n"

        report_content += "=========================================\n"
        print("\n" + report_content)

        report_file = output_dir / "report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 报告已保存至 {report_file}")

    print("\n所有任务处理完毕。")


if __name__ == "__main__":
    main()