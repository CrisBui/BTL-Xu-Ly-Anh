import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import tempfile
import glob
import shutil
from PIL import Image  # Dùng PIL thay cho cv2
import datetime
import random

# Import class từ file logic (Giả định bạn đã thêm generate_random_file_name vào class)
from stitcher_manual import PanoramaStitcher


# --- HÀM TIỆN ÍCH CHUNG ---
def generate_random_file_name():
    """Tạo tên file ngẫu nhiên theo cấu trúc YearMonthDayHourMinuteSecond_Random"""
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_int = random.randint(1000, 9999)
    return f"{timestamp_str}_{random_int}"


# Khởi tạo đối tượng Stitcher một lần
@st.cache_resource
def get_stitcher():
    # Sử dụng generate_random_file_name từ bên ngoài hoặc thêm vào class nếu cần
    return PanoramaStitcher()


stitcher = get_stitcher()


# --- HÀM TIỆN ÍCH CHO XỬ LÝ FILE STREAMLIT ---

def save_uploaded_files_and_get_paths(uploaded_files):
    """Lưu files vào thư mục tạm thời và trả về danh sách đường dẫn đã sắp xếp."""
    temp_dir = tempfile.mkdtemp()
    saved_files = []

    for i, uploaded_file in enumerate(uploaded_files):
        # Đặt tên file mới: "00_timestamp_random.ext" để đảm bảo thứ tự sắp xếp
        original_name = uploaded_file.name
        ext = os.path.splitext(original_name)[1].lower()

        new_file_name = f"{i:02d}_{generate_random_file_name()}{ext}"
        temp_file_path = os.path.join(temp_dir, new_file_name)

        # Lưu file buffer (dùng PIL cho tính đồng nhất)
        img_pil = Image.open(uploaded_file)
        img_pil.save(temp_file_path)

        saved_files.append(temp_file_path)

    return sorted(saved_files), temp_dir


# --- HÀM CHÍNH CỦA STREAMLIT ---

def main():
    st.set_page_config(layout="wide")
    st.title("📸 Panorama Stitching Tool (Manual SIFT)")
    st.markdown("### Bài Tập Lớn Môn Xử Lý Ảnh - Ghép ảnh Panorama Thủ công")
    st.markdown("---")

    # 1. Tải ảnh lên
    uploaded_files = st.sidebar.file_uploader(
        "Tải lên 2-5 ảnh để ghép (Tải theo thứ tự)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if 'current_paths' not in st.session_state:
        st.session_state['current_paths'] = None

    if uploaded_files:
        st.sidebar.success(f"Đã tải lên {len(uploaded_files)} tệp.")

        # Cập nhật đường dẫn khi file được upload
        sorted_paths, temp_dir = save_uploaded_files_and_get_paths(uploaded_files)
        st.session_state['current_paths'] = sorted_paths
        st.session_state['temp_dir'] = temp_dir

        if st.sidebar.button("Bắt đầu Ghép ảnh"):

            paths = st.session_state['current_paths']
            if len(paths) < 2:
                st.error("Cần tối thiểu 2 ảnh hợp lệ.")
                return

            st.markdown("---")
            st.markdown("## ⚙️ 2. Quá trình Xử lý")

            with st.spinner("Đang tính toán SIFT, Homography và Ghép ảnh... (Quá trình có thể mất thời gian)"):

                # --- CHẠY CORE ALGORITHM ---
                # stitcher.stitch: Input list paths, Output là (Ảnh RGB cuối cùng, Dữ liệu viz)
                final_panorama_rgb, matches_pairs, ransac_matched_pairs = stitcher.stitch(paths)

            # --- XÓA FILE TẠM (CLEANUP) ---
            shutil.rmtree(st.session_state['temp_dir'], ignore_errors=True)

            if final_panorama_rgb is None:
                st.error("Lỗi: Quá trình ghép ảnh thất bại (Không đủ khớp hoặc lỗi tính toán).")
                return

            # --- 4. OUTPUT a: Visualization Keypoints and Matches ---
            st.markdown("### 2.1. Visualization Keypoints and Matches")

            for i in range(len(matches_pairs)):
                # st.markdown("---")
                # st.markdown("## ✨ Ảnh Matches")
                #
                # st.image(matches_pairs[i], caption="Ảnh Matches", use_column_width=True)

                st.markdown("---")
                st.markdown("## ✨ Ảnh Matches Ransac")

                st.image(ransac_matched_pairs[i], caption="Ảnh Matches Ransac", use_column_width=True)

            # --- 5. OUTPUT b: Final Panorama ---
            st.markdown("---")
            st.markdown("## ✨ 2.2. Ảnh Panorama Cuối cùng")

            st.image(final_panorama_rgb, caption="Ảnh Panorama Đã Ghép", use_column_width=True)

            img_pil = Image.fromarray(final_panorama_rgb.astype(np.uint8))

            # Lưu PIL Image vào buffer để download
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG")

            st.download_button(
                label="Tải về Ảnh Panorama",
                data=buffer.getvalue(),
                file_name="panorama_final.jpg",
                mime="image/jpeg"
            )

    else:
        st.info("Vui lòng tải lên ít nhất 2 ảnh để bắt đầu.")


if __name__ == "__main__":
    main()