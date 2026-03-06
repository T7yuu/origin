import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt


def save_binary_mask(mask: np.ndarray, path: Path):
    binary_mask_img = (mask * 255).astype(np.uint8)
    Image.fromarray(binary_mask_img).save(path)


def calculate_3d_metrics(pred_volume: np.ndarray, gt_volume: np.ndarray) -> Optional[Dict[str, float]]:

    if pred_volume.shape != gt_volume.shape:
        logging.error(f"预测体数据和标签体数据的尺寸不匹配: {pred_volume.shape} vs {gt_volume.shape}")
        return None

    # 确保是布尔类型
    pred_mask = pred_volume.astype(bool)
    gt_mask = gt_volume.astype(bool)

    epsilon = 1e-8

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    dsc = (2. * intersection) / (pred_sum + gt_sum + epsilon)

    surface_pred = np.logical_xor(pred_mask,
                                  cv2.erode(pred_mask.astype(np.uint8), np.ones((3, 3, 3)), iterations=1).astype(bool))
    surface_gt = np.logical_xor(gt_mask,
                                cv2.erode(gt_mask.astype(np.uint8), np.ones((3, 3, 3)), iterations=1).astype(bool))

    if not np.any(surface_pred) or not np.any(surface_gt):
        nsd = 0.0 if np.array_equal(pred_mask, gt_mask) else 0.0  # 理论上应为1.0，但为避免歧义设为0
    else:
        dist_pred = distance_transform_edt(~surface_pred)
        dist_gt = distance_transform_edt(~surface_gt)

        distances_gt_to_pred = dist_pred[surface_gt]

        distances_pred_to_gt = dist_gt[surface_pred]

        tau = 1
        nsd_gt = np.sum(distances_gt_to_pred <= tau) / len(distances_gt_to_pred)
        nsd_pred = np.sum(distances_pred_to_gt <= tau) / len(distances_pred_to_gt)
        nsd = (nsd_gt * len(distances_gt_to_pred) + nsd_pred * len(distances_pred_to_gt)) / (
                    len(distances_gt_to_pred) + len(distances_pred_to_gt))

    return {"dsc": dsc, "nsd": nsd}

def calculate_metrics_for_folder(pred_dir: str, gt_dir: str) -> (float, float):

    logging.info(f"开始计算评测指标... 预测文件夹: {pred_dir}, 标签文件夹: {gt_dir}")

    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)

    total_dice = 0.0
    total_iou = 0.0
    image_count = 0

    pred_files = [f for f in pred_path.iterdir() if f.is_file()]
    if not pred_files:
        logging.warning("预测文件夹为空，无法计算指标。")
        return 0.0, 0.0

    for pred_file in pred_files:
        gt_file = gt_path / pred_file.name
        if not gt_file.exists():
            logging.warning(f"找不到对应的标签文件: {gt_file}，已跳过。")
            continue

        try:
            pred_mask = np.array(Image.open(pred_file).convert('L')) > 127
            gt_mask = np.array(Image.open(gt_file).convert('L')) > 127

            if pred_mask.shape != gt_mask.shape:
                logging.warning(f"文件 {pred_file.name} 的尺寸与标签不匹配，已跳过。")
                continue

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
            iou = intersection / (union + 1e-8)

            total_dice += dice
            total_iou += iou
            image_count += 1
        except Exception as e:
            logging.error(f"处理文件 {pred_file.name} 进行评测时出错: {e}")
            continue

    if image_count == 0:
        logging.error("没有成功处理任何图片对用于评测。")
        return 0.0, 0.0

    avg_dice = total_dice / image_count
    avg_miou = total_iou / image_count

    logging.info(f"评测完成: 共处理 {image_count} 张图片。")
    logging.info(f"  - 平均 Dice 分数: {avg_dice:.4f}")
    logging.info(f"  - 平均 mIoU 分数: {avg_miou:.4f}")

    return avg_dice, avg_miou


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def visualize_and_save_result(input_image: Image.Image, result_entry: Dict[str, Any], output_path: Path):

    output_path.parent.mkdir(exist_ok=True, parents=True)
    bg_color = (255, 255, 255)
    padding = 25

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arialbd.ttf", 28)
    except IOError:
        font = ImageFont.load_default()
        title_font = font

    img_input = input_image.copy().convert("RGB")

    reference_data = result_entry.get('retrieval', {}).get('data', {})
    if reference_data and Path(reference_data.get('image_path', '')).exists():
        img_ref = Image.open(reference_data['image_path']).convert("RGB")
        draw_ref = ImageDraw.Draw(img_ref)
        rx, ry, rw, rh = [int(c) for c in reference_data.get('box', [0, 0, 0, 0])]
        draw_ref.rectangle([rx, ry, rx + rw, ry + rh], outline="lime", width=5)
    else:
        img_ref = Image.new("RGB", img_input.size, (224, 224, 224))
        ImageDraw.Draw(img_ref).text((10, 10), "No/Invalid Reference", font=font, fill=(0, 0, 0))

    img_final = input_image.copy().convert("RGB")
    final_mask = result_entry.get('final_mask')
    final_box = result_entry.get('final_box')

    if final_mask is not None:
        fig, ax = plt.subplots(figsize=(img_final.width / 100, img_final.height / 100), dpi=100)
        ax.imshow(img_final)

        if isinstance(final_mask, list):
            final_mask = np.array(final_mask)

        show_mask(final_mask, ax, random_color=True)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.draw()

        argb_buffer = fig.canvas.tostring_argb()
        img_array = np.frombuffer(argb_buffer, dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_final = Image.fromarray(img_array.reshape(h, w, 4)).convert('RGB')

        plt.close(fig)
    elif final_box:
        draw_final = ImageDraw.Draw(img_final)
        x, y, w, h = [int(c) for c in final_box]
        draw_final.rectangle([x, y, x + w, y + h], outline="red", width=5)

    images_to_show = [img_input, img_ref, img_final]
    labels = ["Input Image", "Retrieved Reference", "Final Segmentation"]
    max_h = max(img.height for img in images_to_show)

    images_resized = []
    for img in images_to_show:
        ratio = max_h / img.height
        new_w = int(img.width * ratio)
        images_resized.append(img.resize((new_w, max_h), Image.Resampling.LANCZOS))

    total_width = sum(img.width for img in images_resized) + padding * (len(images_resized) + 1)
    text_area_height = 150
    grid_h = max_h + padding * 2 + text_area_height
    grid_img = Image.new('RGB', (total_width, grid_h), bg_color)
    grid_draw = ImageDraw.Draw(grid_img)

    grid_draw.text((padding, 5), f"Result for: '{result_entry.get('file_name', 'N/A')}'", font=title_font,
                   fill=(0, 0, 0))
    current_x = padding
    for i, img in enumerate(images_resized):
        grid_img.paste(img, (current_x, padding + 40))
        grid_draw.text((current_x, padding + 15), labels[i], font=font, fill=(0, 0, 0))
        current_x += img.width + padding

    query_text = result_entry.get('query', 'N/A')
    final_path = result_entry.get('final_path', 'N/A')
    info_text = f"Query: {query_text}\nFinal Path Taken: {final_path}\n"
    if result_entry.get('retrieval'):
        info_text += f"Best match content_dist: {result_entry['retrieval'].get('distance', -1):.4f}, score: {result_entry['retrieval'].get('final_score', -1):.4f}\n"
    if result_entry.get('mask_score'):
        info_text += f"Segmentation Mask Score: {result_entry['mask_score']:.4f}"

    grid_draw.multiline_text((padding, max_h + padding + 60), info_text, font=font, fill=(0, 0, 0))
    grid_img.save(output_path)


def expand_box(box: List[int], image_size: tuple, ratio: float) -> List[int]:
    if ratio <= 1.0:
        return box

    img_w, img_h = image_size
    x, y, w, h = box

    center_x = x + w / 2
    center_y = y + h / 2

    new_w = w * ratio
    new_h = h * ratio

    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2

    final_x = max(0, int(new_x))
    final_y = max(0, int(new_y))

    final_w = min(int(new_w), img_w - final_x)
    final_h = min(int(new_h), img_h - final_y)

    return [final_x, final_y, final_w, final_h]

def generate_box_from_robust_average(retrieved_results: List[Dict], query_image_size: tuple) -> Optional[List[int]]:
    logging.info("触发 Tier 2: 鲁棒平均策略...")
    relative_boxes, box_centers, weights = [], [], []
    for item in retrieved_results:
        try:
            ref_box_abs = item['data']['box']
            with Image.open(item['data']['image_path']) as ref_img:
                ref_img_size = ref_img.size
            box_rel = box_abs_to_rel(ref_box_abs, ref_img_size)
            relative_boxes.append(box_rel)
            box_centers.append([box_rel[0] + box_rel[2] / 2, box_rel[1] + box_rel[3] / 2])
            weights.append(1.0 / (item.get('final_score', 1.0) + 1e-6))
        except Exception as e:
            logging.warning(f"处理参考案例时出错: {e}, 已跳过。")
            continue
    if len(relative_boxes) < 3:
        logging.warning(f"有效的参考案例不足3个({len(relative_boxes)}个)，将使用简单加权平均。")
        if not relative_boxes: return None
        normalized_weights = np.array(weights) / sum(weights)
        avg_rel_box = np.sum(np.array(relative_boxes) * normalized_weights[:, np.newaxis], axis=0)
        return box_rel_to_abs(avg_rel_box.tolist(), query_image_size)
    relative_boxes = np.array(relative_boxes)
    box_centers = np.array(box_centers)
    weights = np.array(weights)
    centroid = np.mean(box_centers, axis=0)
    distances_to_centroid = np.linalg.norm(box_centers - centroid, axis=1)
    median_dist = np.median(distances_to_centroid)
    is_not_outlier = distances_to_centroid < 2.5 * median_dist
    if not np.any(is_not_outlier): is_not_outlier = np.ones_like(is_not_outlier, dtype=bool)
    logging.info(f"离群点检测: 原始样本数 {len(relative_boxes)}, 保留核心样本数 {np.sum(is_not_outlier)}")
    core_boxes = relative_boxes[is_not_outlier]
    core_weights = weights[is_not_outlier]
    normalized_weights = core_weights / np.sum(core_weights)
    avg_rel_box = np.sum(core_boxes * normalized_weights[:, np.newaxis], axis=0)
    logging.info(f"✅ 鲁棒平均计算完成，最终相对Box: {avg_rel_box.tolist()}")
    return box_rel_to_abs(avg_rel_box.tolist(), query_image_size)


def setup_logger(log_path: Path):
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in [file_handler, console_handler]:
        handler.setLevel(logging.INFO);
        handler.setFormatter(formatter);
        logger.addHandler(handler)
    return logger


def box_abs_to_rel(box_abs: List[int], img_size: tuple) -> List[float]:
    img_w, img_h = img_size
    if img_w == 0 or img_h == 0: return [0.0, 0.0, 0.0, 0.0]
    x, y, w, h = box_abs
    return [x / img_w, y / img_h, w / img_w, h / img_h]


def box_rel_to_abs(box_rel: List[float], img_size: tuple) -> List[int]:
    img_w, img_h = img_size
    rel_x, rel_y, rel_w, rel_h = box_rel
    return [int(rel_x * img_w), int(rel_y * img_h), int(rel_w * img_w), int(rel_h * img_h)]