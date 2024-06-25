import argparse
import cv2
import numpy as np
import os
import configparser


def expand_masks(config):
    input_folder = config["expand"]["input"]
    output_folder = config["expand"]["output"]
    # 定义膨胀核
    kernel = np.ones((5, 5), np.uint8)  # 这里的(5, 5)可以根据需求调整膨胀核的大小

    # 确保输出文件夹存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 读取图片
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, 0)  # 以灰度模式读取

            # 进行膨胀操作
            dilated_image = cv2.dilate(image, kernel, iterations=15)

            # 写入膨胀后的图片到输出文件夹
            output_filepath = os.path.join(output_folder, filename)
            cv2.imwrite(output_filepath, dilated_image)

            print(f"{filename} processed.")

    print("All images processed.")


def expand_mask():
    parser = argparse.ArgumentParser(description="Expand mask images")
    # parser.add_argument(
    #     "--config", type=str, required=True, help="Path to the config file"
    # )
    # args = parser.parse_args()
    config_file = "mixConfig.ini"

    config = configparser.ConfigParser()
    config.read(config_file)

    expand_masks(config)
