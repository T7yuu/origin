# test_sam2_image.py

import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 确保SAM 2的模块可以被正确导入
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("导入SAM 2模块失败。请确保您已在sam2项目根目录下，并已成功执行 'pip install -e .'。")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO)


def show_mask(mask, ax, random_color=False):
    """在图像上可视化单个掩码"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # 道奇蓝，带60%透明度
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """在图像上可视化提示点"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# ★ 新增：用于可视化提示框的函数
def show_box(box, ax):
    """在图像上可视化提示框"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def run_prediction(args):
    """主执行函数"""
    logging.info("开始执行SAM 2图片预测...")

    # 1. 加载模型和预测器
    logging.info(f"加载模型权重: {args.checkpoint}")
    logging.info(f"加载模型配置: {args.model_cfg}")

    sam_model = build_sam2(args.model_cfg, args.checkpoint)
    predictor = SAM2ImagePredictor(sam_model)

    predictor.model.to(args.device)
    logging.info(f"模型已加载到设备: {args.device}")

    # 2. 加载并预处理图片
    try:
        image = Image.open(args.image).convert("RGB")
    except FileNotFoundError:
        logging.error(f"错误: 无法找到图片文件 '{args.image}'")
        sys.exit(1)

    # 3. ★★★ 定义您的提示 (Define Your Prompts) ★★★
    # 当前配置为使用“框提示”(Box Prompt)

    # 示例: 框提示 (Box Prompt)
    # 定义一个覆盖图像中心50%区域的框 [x1, y1, x2, y2]
    # 您可以根据需要修改这里的坐标
    image_width, image_height = image.size
    x1 = image_width * 0.25
    y1 = image_height * 0.25
    x2 = image_width * 0.75
    y2 = image_height * 0.75
    input_box = np.array([x1, y1, x2, y2])

    # # 如果需要使用点提示，请注释掉上面的 input_box 并取消下面的注释
    # input_point = np.array([[image_width * 0.5, image_height * 0.5]])
    # input_label = np.array([1])

    logging.info(f"使用提示框: {input_box}")

    # 4. 执行预测
    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
        predictor.set_image(image)

        # ★ 修改：调用 predict 时传入 box 参数
        masks, scores, logits = predictor.predict(
            # point_coords=input_point,  # 已禁用点提示
            # point_labels=input_label,
            box=input_box,  # ★ 激活框提示
            multimask_output=True,
        )

    logging.info("预测完成！正在生成可视化结果...")

    # 5. 可视化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca(), random_color=True)
        logging.info(f"掩码 {i + 1}, 质量分数: {score:.3f}")

    # ★ 修改：显示提示框而不是提示点
    show_box(input_box, plt.gca())

    plt.title(f"SAM 2 预测结果 (使用框提示)")
    plt.axis('off')

    plt.savefig(args.output)
    logging.info(f"结果已保存到: {args.output}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用SAM 2对单张图片进行掩码预测的测试案例。"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=r'D:\Codes\PaperCodes\AAAI_v2\sam2\sam2.1_hiera_base_plus.pt', help="SAM 2模型权重文件 (.pt) 的路径。"
    )
    parser.add_argument(
        "--model-cfg", type=str, default=r'D:\Codes\PaperCodes\AAAI_v2\sam2\config.yaml', help="SAM 2模型配置文件 (.yaml) 的路径。"
    )
    parser.add_argument(
        "--image", type=str, default=r'D:\Codes\PaperCodes\AAAI_v2\sam2\ISIC_0000189.jpg', help="需要进行预测的输入图片路径。"
    )
    parser.add_argument(
        "--output", type=str, default="sam2_output.png", help="保存可视化结果的输出路径。"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="运行设备，'cuda' 或 'cpu'。"
    )
    args = parser.parse_args()
    run_prediction(args)