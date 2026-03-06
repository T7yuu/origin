import os
import cv2
import numpy as np
from tqdm import tqdm


def get_scaled_bbox_from_mask(mask_path, scale_factor=1.1):
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"\n警告：无法读取图片 {os.path.basename(mask_path)}，已跳过。")
            return "null"

        rows, cols = np.where(mask == 255)

        if len(rows) == 0:
            return "null"

        img_h, img_w = mask.shape

        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        new_width = width * scale_factor
        new_height = height * scale_factor

        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2

        final_x_min = max(0, new_x_min)
        final_y_min = max(0, new_y_min)
        final_x_max = min(img_w - 1, new_x_max)
        final_y_max = min(img_h - 1, new_y_max)

        return [int(final_x_min), int(final_y_min), int(final_x_max), int(final_y_max)]

    except Exception as e:
        print(f"\n处理文件 {os.path.basename(mask_path)} 时发生未知错误: {e}")
        return "null"


def process_directory_recursively(src_root, dest_root, scale_factor=1.1):
    if not os.path.isdir(src_root):
        print(f"警告：源文件夹 '{src_root}' 不存在，已跳过。")
        return

    print(f"正在处理任务: '{os.path.basename(src_root)}'")
    print(f"  -> 源目录: {src_root}")
    print(f"  -> 目标目录: {dest_root}")

    dir_walk = list(os.walk(src_root))
    for current_dir, _, files in tqdm(dir_walk, desc="  -> 进度"):

        image_files = [f for f in files if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.nii.gz'))]
        if not image_files:
            continue

        relative_path = os.path.relpath(current_dir, src_root)

        output_dir = os.path.join(dest_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        for filename in image_files:
            mask_path = os.path.join(current_dir, filename)

            result = get_scaled_bbox_from_mask(mask_path, scale_factor)

            if isinstance(result, str):
                bbox_str = result
            else:
                bbox_str = ' '.join(map(str, result))

            base_filename = os.path.splitext(filename)[0]
            if base_filename.lower().endswith('.nii'):
                base_filename = os.path.splitext(base_filename)[0]

            output_path = os.path.join(output_dir, f"{base_filename}.txt")

            with open(output_path, 'w') as f:
                f.write(bbox_str)


if __name__ == "__main__":
    folder_mapping = {
        r"D:\Datasets\MSD\Task02_Heart\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task02_heart",
        r"D:\Datasets\MSD\Task04_Hippocampus\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task04_hippocampus",
        r"D:\Datasets\MSD\Task05_Prostate\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task05_prostate",
        r"D:\Datasets\MSD\Task09_Spleen\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task09_spleen"
    }

    for src, dest in folder_mapping.items():
        process_directory_recursively(src, dest)
        print("-" * 60)

    print("\n所有任务处理完成！")