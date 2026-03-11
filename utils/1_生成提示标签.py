import os
import cv2
import numpy as np
from tqdm import tqdm


def find_and_save_scaled_bbox(src_folder, dest_folder, scale_factor=1.1):
    if not os.path.isdir(src_folder):
        print(f"警告：源文件夹 '{src_folder}' 不存在，已跳过。")
        return

    os.makedirs(dest_folder, exist_ok=True)
    print(f"正在处理文件夹: '{src_folder}'")
    print(f"检测到的边界框将保存至: '{dest_folder}'")

    image_files = [f for f in os.listdir(src_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    if not image_files:
        print("  -> 在此文件夹中未找到图片文件。")
        return

    for filename in tqdm(image_files, desc=f"  -> 进度"):
        mask_path = os.path.join(src_folder, filename)

        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告：无法读取图片 {filename}，已跳过。")
                continue

            img_h, img_w = mask.shape

            rows, cols = np.where(mask == 255)

            if len(rows) == 0:
                continue

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

            bbox = [int(final_x_min), int(final_y_min), int(final_x_max), int(final_y_max)]

            base_filename = os.path.splitext(filename)[0]
            output_path = os.path.join(dest_folder, f"{base_filename}.txt")
            bbox_str = ' '.join(map(str, bbox))

            with open(output_path, 'w') as f:
                f.write(bbox_str)

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")


if __name__ == "__main__":
    '''folder_mapping = {
        r"D:/Datasets/ISIC2018/test/groundtruth": r"D:\Datasets\MedSAM_Prompts\ISIC_2018",
        r"D:\Datasets\BraTS\BraTS_test\masks": r"D:\Datasets\MedSAM_Prompts\BrainTS"
    }'''

    folder_mapping = {
        r"E:/Datasets/ISIC2018/test/groundtruth": r"E:/Datasets/ISIC2018/test/prompts"
    }

    for src, dest in folder_mapping.items():
        find_and_save_scaled_bbox(src, dest)
        print("-" * 50)

    print("\n所有任务处理完成！")
