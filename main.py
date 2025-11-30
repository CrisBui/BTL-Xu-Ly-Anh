import datetime
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from ImageVisualizer import viz
from stitcher_manual import PanoramaStitcher

# --- Cấu hình ---
INPUT_DIR = 'Data'
OUTPUT_FILE = 'Output/final_stitched_manual.jpg'
OUTPUT_DIR = 'Output'

def main():
    # Khởi tạo đối tượng Stitcher
    stitcher = PanoramaStitcher()
    # viz = ImageVisualizer(figsize=(12, 12))  # Khởi tạo Visualizer

    image_paths = getImagePaths(INPUT_DIR)

    print(f"Bắt đầu ghép {len(image_paths)} ảnh thủ công...")

    # --- CHẠY PIPELINE GHÉP ẢNH ---
    final_panorama = stitcher.stitch(image_paths)

    if final_panorama is not None:
        # Lưu ảnh cuối cùng
        checkDir(OUTPUT_DIR)

        outputFile = os.path.join(OUTPUT_DIR, generateRandomFileName())
        stitcher.save_image(final_panorama, outputFile)
        print("------------------------------------------------")
        print(f"✅ Hoàn thành ghép ảnh! Kết quả lưu tại: {outputFile}")

        viz.show_final_panorama(final_panorama)
    else:
        print("❌ Ghép ảnh thất bại. Kiểm tra số lượng ảnh (cần >= 2) hoặc chất lượng khớp.")

def checkDir(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

def getImagePaths(inputDir):
    if not os.path.exists(inputDir):
        raise ValueError("Input Dir not found!")

    imgList = sorted(os.listdir(inputDir))  # sorted by name
    image_paths = [os.path.join(inputDir, f) for f in imgList]
    return image_paths

def generateRandomFileName():
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_int = random.randint(100000, 999999)
    return f"{timestamp_str}_{random_int}.jpg"

main()
# 1. Khởi tạo đối tượng
# stitcher = PanoramaStitcher()
#
# img_rgb = stitcher.load_image('TempDatas/tajm1.jpg')
# img_l_small, scale_l = stitcher.resize_image(img_rgb, 600 )
#
# kp1, des1 = stitcher.compute_sift(stitcher.to_gray(img_l_small))
#
# kp = kp1 * (1.0/scale_l)
#
# # Lệnh Test 1: Hiển thị Keypoint
# viz.draw_keypoints(
#     image_rgb=img_rgb,
#     keypoints=kp,
#     title="Test keypoints"
# )