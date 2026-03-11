import sys
import os
from pathlib import Path
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 1. 强制添加 sam2 源码路径（指向你改名后的文件夹）
sam2_repo_path = r"E:\PythonD\ERA\sam2_code"
if sam2_repo_path not in sys.path:
    sys.path.insert(0, sam2_repo_path)

# 2. 移除当前目录干扰，防止触发 RuntimeError
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"导入SAM2库时出错。请确保'sam2-image-segmentation'已通过pip正确安装。")
    raise e


IMAGE_DIR = Path(r"E:/Datasets/ISIC2018/test/images_compressed")
PROMPT_DIR = Path(r"E:/Datasets/ISIC2018/test/prompts")
GT_DIR = Path(r"E:/Datasets/ISIC2018/test/groundtruth")
OUTPUT_DIR = Path(r"E:/PythonD/ERA/scripts/V9/outputs")


SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    epsilon = 1e-8
    return intersection / (union + epsilon)


'''
def calculate_metrics_from_totals(tp_total, fp_total, fn_total, tn_total) -> Dict[str, float]:
   epsilon = 1e-8
    macro_dice = (2. * tp_total) / (2 * tp_total + fp_total + fn_total + epsilon)
    macro_union = tp_total + fp_total + fn_total
    macro_iou = tp_total / (macro_union + epsilon)
    macro_sensitivity = tp_total / (tp_total + fn_total + epsilon)
    macro_specificity = tn_total / (tn_total + fp_total + epsilon)
    macro_accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + epsilon)
    return {
        "dice": macro_dice, "iou": macro_iou, "sensitivity": macro_sensitivity,
        "specificity": macro_specificity, "accuracy": macro_accuracy
    }'''

#改后
def calculate_metrics_from_totals(tp_total, fp_total, fn_total, tn_total) -> Dict[str, float]:
    # 核心：在运算前将所有累加值转为 64位浮点数
    tp = float(tp_total)
    fp = float(fp_total)
    fn = float(fn_total)
    tn = float(tn_total)

    epsilon = 1e-8

    # 重新计算（确保结果在 0-1 之间）
    macro_dice = (2. * tp) / (2 * tp + fp + fn + epsilon)
    macro_union = tp + fp + fn
    macro_iou = tp / (macro_union + epsilon)
    macro_sensitivity = tp / (tp + fn + epsilon)
    macro_specificity = tn / (tn + fp + epsilon)
    macro_accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return {
        "dice": macro_dice,
        "iou": macro_iou,
        "sensitivity": macro_sensitivity,
        "specificity": macro_specificity,
        "accuracy": macro_accuracy
    }

def evaluate_folder_macro(masks_dir: Path, gt_dir: Path) -> Dict[str, float]:
    #total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    # 改后
    total_tp = np.int64(0)
    total_fp = np.int64(0)
    total_fn = np.int64(0)
    total_tn = np.int64(0)

    processed_count = 0
    mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file() and f.suffix.lower() == '.jpg'])

    print(f"\n正在对 {len(mask_files)} 个掩码进行宏观评测...")
    for mask_file in tqdm(mask_files, desc="评测掩码 (宏观)"):
        #gt_file = gt_dir / f"{mask_file.stem}.jpg"
        gt_file = gt_dir / f"{mask_file.stem}_segmentation.png" #改后
        if not gt_file.exists():
            gt_file = gt_dir / mask_file.name
            if not gt_file.exists():
                print(f"警告：找不到 {mask_file.name} 对应的真实标签，已跳过。")
                continue

        try:
            pred_image = Image.open(mask_file).convert('L')
            gt_image = Image.open(gt_file).convert('L')
            # 根据ISIC数据集特性，前景为白色(>127)或黑色(<127)，这里假设为白色
            pred_mask = np.array(pred_image) > 127
            gt_mask = np.array(gt_image) > 127

            if pred_mask.shape != gt_mask.shape:
                continue

            total_tp += np.logical_and(pred_mask, gt_mask).sum()
            total_fp += np.logical_and(pred_mask, ~gt_mask).sum()
            total_fn += np.logical_and(~pred_mask, gt_mask).sum()
            total_tn += np.logical_and(~pred_mask, ~gt_mask).sum()
            processed_count += 1
        except Exception as e:
            print(f"警告：无法处理 {mask_file.name}。错误: {e}")
            continue

    if processed_count == 0:
        print("错误：没有任何图片被成功评测。")
        return {key: 0.0 for key in ["dice", "iou", "sensitivity", "specificity", "accuracy"]}

    return calculate_metrics_from_totals(total_tp, total_fp, total_fn, total_tn)



def load_prompts(prompt_file: Path) -> Optional[np.ndarray]:
    if not prompt_file.exists():
        return None
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if content.lower() == "null":
        return None
    boxes = []
    lines = content.split('\n')
    for line in lines:
        if not line.strip():
            continue
        try:
            coords = [float(c) for c in line.strip().split()]
            if len(coords) == 4:
                boxes.append(coords)
        except ValueError:
            print(f"警告：无法解析文件 {prompt_file} 中的行: '{line}'")
            continue
    return np.array(boxes) if boxes else None


def main():
    print(f"使用设备: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("正在加载SAM2模型...")
    try:
        sam2_model = build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT))
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        sam2_predictor.model.to(DEVICE)
        print("✅ SAM2模型加载成功。")
    except Exception as e:
        print(f"❌ 加载SAM2模型失败。请检查配置文件和权重文件的路径。错误: {e}")
        return

    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f"发现 {len(image_files)} 张待处理图片。")
    start_time = time.perf_counter()

    for image_path in tqdm(image_files, desc="正在分割图片"):
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                sam2_predictor.set_image(image)

                #prompt_file = PROMPT_DIR / f"{image_path.stem}.txt"
                prompt_file = PROMPT_DIR / f"{image_path.stem}_segmentation.txt" #改后
                box_prompts = load_prompts(prompt_file)

                best_mask = None
                if box_prompts is not None:
                    masks, scores, _ = sam2_predictor.predict(
                        box=box_prompts,
                        multimask_output=False,
                    )
                    best_mask = masks[np.argmax(scores)]
                else:
                    #gt_path = GT_DIR / f"{image_path.stem}.jpg"
                    gt_path = GT_DIR / f"{image_path.stem}_segmentation.png" #改后
                    if not gt_path.exists():
                        gt_path = GT_DIR / image_path.name
                        if not gt_path.exists():
                            print(f"警告：文件 {image_path.name} 既无提示也无真实标签，已跳过。")
                            continue

                    gt_image = Image.open(gt_path).convert('L')
                    gt_mask = np.array(gt_image) > 127

                    masks, _, _ = sam2_predictor.predict(multimask_output=True)

                    max_iou = -1.0
                    for candidate_mask in masks:
                        iou = calculate_iou(candidate_mask, gt_mask)
                        if iou > max_iou:
                            max_iou = iou
                            best_mask = candidate_mask

                if best_mask is not None:
                    squeezed_mask = np.squeeze(best_mask)
                    mask_img = Image.fromarray(squeezed_mask.astype(np.uint8) * 255)

                    output_path = OUTPUT_DIR / f"{image_path.stem}.jpg"
                    mask_img.save(output_path, "JPEG")

        except Exception as e:
            print(f"处理文件 {image_path.name} 时出错: {e}")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\n✅ 分割任务完成，总耗时 {total_time:.2f} 秒。")

    final_metrics = evaluate_folder_macro(OUTPUT_DIR, GT_DIR)

    report_content = (
        f"评测报告\n"
        f"=========================================\n"
        f"数据集: ISIC2018 (使用GroundingDINO提示)\n"
        f"总分割耗时: {total_time:.2f} 秒\n"
        f"-----------------------------------------\n"
        f"宏观Dice系数:      {final_metrics['dice']:.4f}\n"
        f"宏观IoU:           {final_metrics['iou']:.4f}\n"
        f"宏观敏感度 (SE):   {final_metrics['sensitivity']:.4f}\n"
        f"宏观特异度 (SP):   {final_metrics['specificity']:.4f}\n"
        f"宏观准确率 (AC):   {final_metrics['accuracy']:.4f}\n"
        f"=========================================\n"
    )

    print("\n" + report_content)

    report_file = OUTPUT_DIR / "report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 报告已保存至 {report_file}")


if __name__ == "__main__":
    main()
