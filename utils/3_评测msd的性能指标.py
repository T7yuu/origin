import numpy as np
from pathlib import Path
from PIL import Image
import logging

def setup_logger(log_file_path: Path):
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

def compute_dsc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection) / total


def compute_nsd(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), tau=2.0) -> float:
    try:
        from medpy.metric.binary import __surface_distances
    except ImportError:
        logging.warning("medpy 未安装，NSD 无法计算，返回 -1")
        return -1.0

    def _nsd(pred, gt, voxelspacing=None, tau=2.0):
        if not np.any(pred) and not np.any(gt):
            return 1.0
        if not np.any(pred) or not np.any(gt):
            return 0.0
        distances_pred_gt = __surface_distances(pred, gt, voxelspacing)
        distances_gt_pred = __surface_distances(gt, pred, voxelspacing)
        nsd = (distances_pred_gt <= tau).sum() + (distances_gt_pred <= tau).sum()
        total_surface = len(distances_pred_gt) + len(distances_gt_pred)
        return nsd / total_surface if total_surface > 0 else 0.0

    return _nsd(pred, gt, voxelspacing=spacing, tau=tau)

def main():
    predicted_masks_dir = Path(r"D:\Datasets\MedSAM_results\MSD\task09_spleen")
    ground_truth_masks_dir = Path(r"D:\Datasets\MSD\Task09_Spleen\labelsTr")
    log_file_path = Path("logs/evaluation.log")

    setup_logger(log_file_path)

    logging.info(f"开始评估：预测路径 = {predicted_masks_dir}")
    logging.info(f"真值路径 = {ground_truth_masks_dir}")

    if not predicted_masks_dir.exists():
        logging.error("预测掩模目录不存在！")
        return
    if not ground_truth_masks_dir.exists():
        logging.error("真值掩模目录不存在！")
        return

    all_dsc = []
    all_nsd = []

    case_dirs = [d for d in predicted_masks_dir.iterdir() if d.is_dir()]
    logging.info(f"共找到 {len(case_dirs)} 个预测病例子文件夹")

    for case_dir in sorted(case_dirs):
        case_name = case_dir.name
        gt_case_dir = ground_truth_masks_dir / case_name
        pred_files = sorted([f for f in case_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        if not pred_files:
            logging.info(f"跳过病例 '{case_name}'：预测文件夹为空")
            continue

        if not gt_case_dir.exists():
            logging.warning(f"跳过病例 '{case_name}'：真值子文件夹不存在")
            continue

        pred_masks = []
        gt_masks = []

        for pred_file in pred_files:
            gt_file = gt_case_dir / pred_file.name
            if not gt_file.exists():
                logging.debug(f"跳过切片 '{pred_file.name}'：真值文件不存在")
                continue

            try:
                pred_img = Image.open(pred_file).convert('L')
                gt_img = Image.open(gt_file).convert('L')
                pred_mask = np.array(pred_img) > 127
                gt_mask = np.array(gt_img) > 127


                pred_masks.append(pred_mask)
                gt_masks.append(gt_mask)
            except Exception as e:
                logging.error(f"读取掩模失败：{pred_file} 或 {gt_file}，错误：{e}")
                continue

        if not pred_masks or len(pred_masks) != len(gt_masks):
            logging.info(f"跳过病例 '{case_name}'：无有效掩模对")
            continue

        try:
            pred_vol = np.stack(pred_masks, axis=0)
            gt_vol = np.stack(gt_masks, axis=0)
        except Exception as e:
            logging.error(f"构建 3D 体积失败（病例 {case_name}）：{e}")
            continue

        try:
            dsc = compute_dsc(pred_vol, gt_vol)
            nsd = compute_nsd(pred_vol, gt_vol)
            all_dsc.append(dsc)
            all_nsd.append(nsd)
            logging.info(f"✅ 病例 '{case_name}' 评估完成: DSC={dsc:.4f}, NSD={nsd:.4f}")
        except Exception as e:
            logging.error(f"计算指标失败（病例 {case_name}）：{e}")
            continue

    num_cases = len(all_dsc)
    avg_dsc = np.mean(all_dsc) if num_cases > 0 else -1.0
    avg_nsd = np.mean(all_nsd) if num_cases > 0 else -1.0

    report = (
        f"\n{'='*50}\n"
        f"3D 分割性能评估报告\n"
        f"{'='*50}\n"
        f"有效病例数量: {num_cases}\n"
        f"平均 DSC: {avg_dsc:.4f}\n"
        f"平均 NSD: {avg_nsd:.4f}\n"
        f"{'='*50}\n"
    )
    print(report)
    logging.info("评估完成。")

    report_path = Path("results/3d_metrics_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logging.info(f"报告已保存至: {report_path}")

if __name__ == "__main__":
    main()