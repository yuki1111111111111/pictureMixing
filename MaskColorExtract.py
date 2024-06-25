import cv2
import os
import numpy as np
import argparse
import configparser
import subprocess


def extract_color_parts(config):
    # 讀取圖片的路徑
    mask_folder = config["extract"]["mask_folder"]
    image_folder = config["extract"]["image_folder"]
    result_folder = config["extract"]["result_folder"]

    # 獲取圖片名稱列表
    mask_files = os.listdir(mask_folder)
    image_files = os.listdir(image_folder)

    # 確保遮罩和圖片數量相同
    assert len(mask_files) == len(
        image_files
    ), "Number of masks and images must be the same"

    # 遍歷每一個遮罩和圖片
    for mask_file, image_file in zip(mask_files, image_files):
        # 讀取遮罩和圖片
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(os.path.join(image_folder, image_file))

        # 確保遮罩和圖片的尺寸相同
        assert (
            mask.shape == image.shape[:2]
        ), "Mask and image must have the same dimensions"

        # 創建一個全黑的圖片
        result = np.zeros_like(image)

        # 將遮罩區域的圖片保留下來
        result[mask == 255] = image[mask == 255]

        # 保存結果圖片
        cv2.imwrite(os.path.join(result_folder, "result_" + image_file), result)


def MaskColorExtract():
    parser = argparse.ArgumentParser(description="Extract color parts using mask")
    config_file = "mixConfig.ini"

    config = configparser.ConfigParser()
    config.read(config_file)

    extract_color_parts(config)
    print("All images processed.")
