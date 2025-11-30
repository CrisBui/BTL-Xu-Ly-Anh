import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

class ImageVisualizer:

    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize

    def draw_keypoints(self, image_rgb, keypoints, title="Detected Keypoints"):
        """
        Vẽ các Keypoint (tọa độ X, Y) lên ảnh.
        image_rgb: ảnh gốc (NumPy array, phải là RGB)
        keypoints: mảng NumPy 2D chứa tọa độ [[x1, y1], [x2, y2], ...] (phải là [X, Y])
        """
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            print("LỖI VIZ: Keypoints phải là mảng N x 2.")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        # Matplotlib cần ảnh RGB và phải là uint8
        ax.imshow(image_rgb.astype(np.uint8))

        # Tọa độ Keypoint (X là cột, Y là hàng)
        keypoint_x = keypoints[:, 0]
        keypoint_y = keypoints[:, 1]

        ax.scatter(keypoint_x, keypoint_y,
                   s=30,
                   c='red',  # Chọn màu nổi bật hơn
                   marker='o',
                   alpha=0.7)

        ax.set_title(title)
        ax.axis('off')
        plt.show()
        plt.close(fig)

    def draw_matches_manual(self, img_left_rgb, kp_left, img_right_rgb, kp_right, matches, title="Feature Matches"):
        """
        Hiển thị hai ảnh cạnh nhau và nối các cặp Keypoint tương ứng.
        kp_left, kp_right PHẢI là tọa độ đã được SCALED NGƯỢC (trên ảnh gốc).
        """
        h_l, w_l = img_left_rgb.shape[:2]
        h_r, w_r = img_right_rgb.shape[:2]

        # Tạo canvas lớn hơn để chứa cả hai ảnh cạnh nhau
        max_h = max(h_l, h_r)
        combined_img = np.zeros((max_h, w_l + w_r, 3), dtype=np.uint8)

        # Đặt ảnh trái và phải lên canvas
        combined_img[:h_l, :w_l] = img_left_rgb
        combined_img[:h_r, w_l:] = img_right_rgb

        # Setup Figure
        fig, ax = plt.subplots(figsize=(self.figsize[0] * 3, self.figsize[1]))  # Giả định self.figsize tồn tại
        ax.imshow(combined_img)

        # Vẽ các đường khớp
        for idx_r, idx_l in matches:
            pt_r = kp_right[idx_r]
            pt_l = kp_left[idx_l]

            # Điều chỉnh tọa độ X của ảnh phải
            pt_r_shifted = (pt_r[0] + w_l, pt_r[1])

            ax.plot([pt_l[0], pt_r_shifted[0]], [pt_l[1], pt_r_shifted[1]],
                    color='lime',
                    linestyle='-',
                    linewidth=0.5,
                    marker='.',
                    markersize=2)

        ax.set_title(f"{title} ({len(matches)} Matches)")
        ax.axis('off')
        plt.show()
        plt.close(fig)

    def draw_keypoint_and_matches_unified(self, img_left_rgb, kp_left_small, scale_left,
                                          img_right_rgb, kp_right_small, scale_right,
                                          matches, title="Unified Keypoint and Match Visualization"):
        """
        Hiển thị cả Keypoint (điểm đỏ) và Đường khớp (đường xanh) trên hai ảnh tách biệt.

        Hàm này tự động tính toán tọa độ Keypoint gốc bằng cách sử dụng hệ số scale.

        img_left_rgb, img_right_rgb: Ảnh gốc (lớn) RGB.
        kp_left_small, kp_right_small: Tọa độ Keypoint trên ảnh đã resize (ảnh nhỏ).
        scale_left, scale_right: Hệ số scale đã dùng để resize ảnh (W_new / W_orig).
        matches: Danh sách các cặp chỉ số khớp (idx_r, idx_l).
        """

        # 1. TÍNH TOÁN ÁNH XẠ NGƯỢC (Mapping Keypoints back to Original Image Size)

        # Tính toán tọa độ Keypoint gốc (Original Coordinates)
        kp_left_original = kp_left_small * (1.0 / scale_left)
        kp_right_original = kp_right_small * (1.0 / scale_right)

        # 2. Định nghĩa khoảng cách và Canvas
        GAP_WIDTH = 100

        h_l, w_l = img_left_rgb.shape[:2]
        h_r, w_r = img_right_rgb.shape[:2]

        max_h = max(h_l, h_r)
        combined_w = w_l + w_r + GAP_WIDTH

        combined_img = np.zeros((max_h, combined_w, 3), dtype=np.uint8)
        x_offset_r = w_l + GAP_WIDTH

        # Đặt ảnh trái và phải lên canvas
        combined_img[:h_l, :w_l] = img_left_rgb
        combined_img[:h_r, x_offset_r:x_offset_r + w_r] = img_right_rgb

        # 3. Setup Figure
        fig, ax = plt.subplots(figsize=(self.figsize[0] * 4, self.figsize[1]))
        ax.imshow(combined_img.astype(np.uint8))

        # --- VẼ KEYPOINT ĐƠN LẺ (Điểm đỏ) ---

        # Ảnh trái (Không dịch chuyển)
        ax.scatter(kp_left_original[:, 0], kp_left_original[:, 1],
                   s=20, c='red', marker='o', alpha=0.9, linewidths=0.5, edgecolors='black')

        # Ảnh phải (Dịch chuyển theo x_offset_r)
        ax.scatter(kp_right_original[:, 0] + x_offset_r, kp_right_original[:, 1],
                   s=20, c='red', marker='o', alpha=0.9, linewidths=0.5, edgecolors='black')

        # --- VẼ ĐƯỜNG KHỚP (Đường xanh) ---

        for idx_r, idx_l in matches:
            pt_r = kp_right_original[idx_r]  # Tọa độ gốc
            pt_l = kp_left_original[idx_l]  # Tọa độ gốc

            # Tọa độ điểm cuối (Ảnh phải) phải được dịch chuyển
            pt_r_shifted_x = pt_r[0] + x_offset_r

            ax.plot([pt_l[0], pt_r_shifted_x], [pt_l[1], pt_r[1]],
                    color='lime',
                    linestyle='-',
                    linewidth=0.8,
                    alpha=0.6)

        # --- HIỂN THỊ CUỐI CÙNG ---
        # ax.set_title(f"{title} ({len(matches)} Khớp)", fontsize=16)
        # ax.axis('off')
        # plt.tight_layout()
        # plt.show()
        # plt.close(fig)

        # --- ĐẦU RA (Thay đổi ở đây) ---
        ax.set_title(f"{title} ({len(matches)} Khớp)", fontsize=16)
        ax.axis('off')
        plt.tight_layout()

        # KHÔNG DÙNG plt.show() HOẶC plt.close() NỮA

        # 1. Lưu figure vào buffer BytesIO
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)

        # 2. Đọc buffer bằng PIL và chuyển thành NumPy array (RGB)
        img_pil = Image.open(buf).convert('RGB')

        # 3. Đóng figure và buffer
        plt.close(fig)
        buf.close()

        return np.array(img_pil)  # Trả về mảng NumPy RGB

    def draw_matches_after_ransac(self, img_left_rgb, kp_left_small, scale_left,
                                  img_right_rgb, kp_right_small, scale_right,
                                  matches, H, threshold=5.0, title="Matches After RANSAC"):
        """
        Hiển thị Keypoint và Đường khớp, phân loại Inliers và Outliers dựa trên ma trận H.

        matches: Danh sách các cặp chỉ số khớp ban đầu (idx_r, idx_l)
        H: Ma trận Homography 3x3 tốt nhất từ RANSAC.
        threshold: Ngưỡng RANSAC (ví dụ: 5.0).
        """

        # 1. TÍNH TOÁN ÁNH XẠ NGƯỢC
        kp_left_original = kp_left_small * (1.0 / scale_left)
        kp_right_original = kp_right_small * (1.0 / scale_right)

        # 2. KIỂM TRA TẤT CẢ CÁC ĐIỂM KHỚP BẰNG H

        # Chuẩn bị dữ liệu cho RANSAC check
        src_pts = np.float32([kp_right_original[m[0]] for m in matches])
        dst_pts = np.float32([kp_left_original[m[1]] for m in matches])

        n_points = len(src_pts)
        src_h = np.hstack([src_pts, np.ones((n_points, 1))])

        # Chiếu điểm nguồn lên điểm đích bằng H
        projected = np.dot(src_h, H.T)
        projected = projected[:, :2] / projected[:, 2:]

        # Tính toán khoảng cách và tìm Inliers
        dists = np.linalg.norm(dst_pts - projected, axis=1)
        # is_inlier là mảng boolean: True nếu là inlier
        is_inlier = dists < threshold

        # 3. Định nghĩa Canvas (Giống hệt hàm cũ)
        GAP_WIDTH = 100
        h_l, w_l = img_left_rgb.shape[:2]
        h_r, w_r = img_right_rgb.shape[:2]
        max_h = max(h_l, h_r)
        combined_w = w_l + w_r + GAP_WIDTH
        combined_img = np.zeros((max_h, combined_w, 3), dtype=np.uint8)
        x_offset_r = w_l + GAP_WIDTH
        combined_img[:h_l, :w_l] = img_left_rgb
        combined_img[:h_r, x_offset_r:x_offset_r + w_r] = img_right_rgb

        # 4. Setup Figure
        fig, ax = plt.subplots(figsize=(self.figsize[0] * 4, self.figsize[1]))
        ax.imshow(combined_img.astype(np.uint8))

        # --- VẼ ĐƯỜNG KHỚP ---
        inlier_count = 0

        for i, (idx_r, idx_l) in enumerate(matches):
            pt_r = kp_right_original[idx_r]
            pt_l = kp_left_original[idx_l]
            pt_r_shifted_x = pt_r[0] + x_offset_r

            if is_inlier[i]:
                # INLIERS: Dùng màu xanh lá cây đậm (ví dụ)
                color = 'green'
                alpha = 1.0
                linewidth = 1.5
                inlier_count += 1
            else:
                # OUTLIERS: Dùng màu xanh lam nhạt (ví dụ)
                color = 'red'
                alpha = 1.0
                linewidth = 1.5

            ax.plot([pt_l[0], pt_r_shifted_x], [pt_l[1], pt_r[1]],
                    color=color,
                    linestyle='-',
                    linewidth=linewidth,
                    alpha=alpha)

        # --- VẼ KEYPOINT ĐƠN LẺ --- (Có thể dùng màu vàng/đỏ cho Keypoint)
        ax.scatter(kp_left_original[:, 0], kp_left_original[:, 1],
                   s=15, c='yellow', marker='o', alpha=0.9, linewidths=0.5, edgecolors='black')
        ax.scatter(kp_right_original[:, 0] + x_offset_r, kp_right_original[:, 1],
                   s=15, c='yellow', marker='o', alpha=0.9, linewidths=0.5, edgecolors='black')

        # --- ĐẦU RA (Tương tự hàm cũ) ---
        ax.set_title(f"{title} ({inlier_count}/{len(matches)} Inliers)", fontsize=16)
        ax.axis('off')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img_pil = Image.open(buf).convert('RGB')
        plt.close(fig)
        buf.close()

        return np.array(img_pil)

    def draw_test_match(self, img_left_rgb, kp_left_small, scale_left,
                        img_right_rgb, kp_right_small, scale_right,
                        matches,
                        title="Paired Keypoint Comparison"):
        """
        Hiển thị hai ảnh cạnh nhau với các Keypoint và đường khớp đã ánh xạ ngược về kích thước gốc.
        """

        # Lấy kích thước ảnh gốc
        h_l, w_l = img_left_rgb.shape[:2]
        h_r, w_r = img_right_rgb.shape[:2]

        # 1. Ánh xạ ngược tọa độ Keypoint về kích thước ảnh GỐC
        scale_factor_l = 1.0 / scale_left
        kp_left_original = kp_left_small * scale_factor_l

        scale_factor_r = 1.0 / scale_right
        kp_right_original = kp_right_small * scale_factor_r

        # 2. Setup Matplotlib Figure với 1 hàng, 2 cột
        # figsize được nhân 2 cho chiều rộng để chứa cả hai ảnh
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 2, self.figsize[1]))

        # Kích thước ảnh gốc
        max_h = max(h_l, h_r)

        # --- CÀI ĐẶT CÁC AXES VÀ VẼ KEYPOINTS ---

        # --- VẼ ẢNH TRÁI (axes[0]) ---
        axes[0].imshow(img_left_rgb.astype(np.uint8))
        axes[0].scatter(kp_left_original[:, 0], kp_left_original[:, 1],
                        s=30, c='red', marker='o', alpha=0.7)
        axes[0].set_title(f"Left Image KPs ({len(kp_left_original)})")
        axes[0].axis('off')

        # --- VẼ ẢNH PHẢI (axes[1]) ---
        axes[1].imshow(img_right_rgb.astype(np.uint8))
        axes[1].scatter(kp_right_original[:, 0], kp_right_original[:, 1],
                        s=30, c='red', marker='o', alpha=0.7)
        axes[1].set_title(f"Right Image KPs ({len(kp_right_original)})")
        axes[1].axis('off')

        # --- VẼ ĐƯỜNG KHỚP (Drawing Match Lines) ---

        # Dùng fig.canvas.figure.transFigure để vẽ đường khớp trên toàn bộ figure
        # LƯU Ý: Đây là cách phức tạp. Cách đơn giản là VẼ TRÊN AXES ĐẦU TIÊN (axes[0])
        # và ánh xạ điểm từ axes[1] về hệ tọa độ của axes[0].
        # TỐT HƠN là vẽ bằng cách sử dụng TỌA ĐỘ VĂN BẢN (DATA COORDINATES)

        # Lặp qua các khớp và vẽ trực tiếp lên hai Axes
        for idx_r, idx_l in matches:
            pt_r = kp_right_original[idx_r]
            pt_l = kp_left_original[idx_l]

            # 1. Chuyển đổi tọa độ Keypoint về tọa độ hiển thị (figure coordinates)
            # Tọa độ X và Y của pt_l nằm trên axes[0]
            # Tọa độ X và Y của pt_r nằm trên axes[1]

            # Lấy tọa độ hiển thị trên figure (display coordinates)
            # Sử dụng transform để chuyển đổi từ data coordinate (pixel) sang figure coordinate

            # Điểm trái (axes[0])
            # [0, 0] vì ta đang vẽ trên axes[0], tọa độ X=pt_l[0], Y=pt_l[1]

            # Điểm phải (axes[1])
            # [1, 0] vì ta đang vẽ trên axes[1], tọa độ X=pt_r[0], Y=pt_r[1]

            # VẼ ĐƯỜNG KHỚP: Cách đơn giản nhất (không hoàn hảo nhưng dễ debug) là
            # chuyển tất cả về Tọa độ Figure và vẽ bằng plt.figure(plot).
            # Tuy nhiên, ta đang dùng Subplots, nên ta sẽ dùng AxesToFigure transformer

            # --- FIX: Sử dụng fig.add_axes để vẽ đường khớp lên toàn bộ vùng ---
            # (Cách này thường quá phức tạp cho BTL)

            # --- FIX TỐT NHẤT: VẼ BẰNG CÁCH NỐI HAI ĐIỂM TRỰC TIẾP TRÊN MÀN HÌNH ---
            # Vấn đề là: Không thể vẽ đường nối giữa hai subplot một cách trực tiếp
            # bằng plot() trên Axes.

            # Bắt buộc phải sử dụng hàm phụ trợ vẽ đường nối cho Subplots,
            # hoặc gộp ảnh. Do đó, chúng ta sẽ quay lại cách GỘP ẢNH TẠM THỜI (giống cv2.drawMatches)

        # --- KHỐI LỆNH TRỰC QUAN HÓA CUỐI CÙNG (Thay thế cho vòng lặp lỗi) ---

        # Để vẽ đường nối giữa 2 subplot, Matplotlib yêu cầu vẽ trên một trục chung
        # hoặc trục ẩn. Thay vì sửa lỗi phức tạp này, chúng ta sẽ áp dụng kỹ thuật
        # GỘP ẢNH TẠM THỜI (gần giống logic trong draw_matches_manual trước đó)
        # và hiển thị nó.

        # Nếu bạn muốn giữ Subplot: Hủy bỏ vòng lặp plot() và chỉ hiển thị điểm.
        # Nếu bạn muốn giữ Match Lines: BẮT BUỘC phải tạo ảnh ghép tạm thời.

        # --- ÁP DỤNG LẠI LOGIC GỘP ẢNH TẠM THỜI CHO MATCH LINES ---

        # 1. Tạo ảnh ghép BGR (giống logic cv2.drawMatches)
        max_h = max(h_l, h_r)
        combined_img = np.zeros((max_h, w_l + w_r, 3), dtype=np.uint8)
        combined_img[:h_l, :w_l] = img_left_rgb
        combined_img[:h_r, w_l:] = img_right_rgb

        # 2. Setup Figure MỚI (chỉ một hình)
        fig_match, ax_match = plt.subplots(figsize=(self.figsize[0] * 3, self.figsize[1]))
        ax_match.imshow(combined_img.astype(np.uint8))

        # 3. Vẽ các đường khớp lên ảnh ghép
        for idx_r, idx_l in matches:
            pt_r = kp_right_original[idx_r]
            pt_l = kp_left_original[idx_l]

            # Điều chỉnh tọa độ X của ảnh phải
            pt_r_shifted_x = pt_r[0] + w_l

            ax_match.plot([pt_l[0], pt_r_shifted_x], [pt_l[1], pt_r[1]],
                          color='lime',
                          linestyle='-',
                          linewidth=0.5,
                          marker='.',
                          markersize=2)

        ax_match.set_title(f"{title} ({len(matches)} Matches)")
        ax_match.axis('off')
        plt.show()
        plt.close(fig_match)

        # --- KẾT THÚC VÒNG LẶP SỬA LỖI (Vòng lặp Match Lines) ---

        # --- HIỂN THỊ CHUNG CHO SUBPLOT KEYPOINTS (giữ nguyên) ---
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Đóng figure Keypoint Subplots

    def draw_paired_keypoints(self, img_left_rgb, kp_left_small, scale_left,
                              img_right_rgb, kp_right_small, scale_right,
                              title="Paired Keypoint Comparison"):
        """
        Hiển thị hai ảnh cạnh nhau với các Keypoint đã ánh xạ ngược về kích thước gốc.

        viz_obj: Đối tượng ImageVisualizer để sử dụng figsize.
        img_left_rgb, img_right_rgb: Ảnh gốc (lớn) RGB.
        kp_left_small, kp_right_small: Tọa độ Keypoint trên ảnh đã resize.
        scale_left, scale_right: Hệ số scale đã dùng để resize ảnh.
        """

        # 1. Ánh xạ ngược tọa độ Keypoint về kích thước ảnh GỐC
        scale_factor_l = 1.0 / scale_left
        kp_left_original = kp_left_small * scale_factor_l

        scale_factor_r = 1.0 / scale_right
        kp_right_original = kp_right_small * scale_factor_r

        # 2. Setup Matplotlib Figure với 1 hàng, 2 cột
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 2, self.figsize[1]))

        # --- VẼ ẢNH TRÁI ---
        axes[0].imshow(img_left_rgb.astype(np.uint8))

        # Vẽ Keypoint đã Scale
        axes[0].scatter(kp_left_original[:, 0], kp_left_original[:, 1],
                        s=30, c='red', marker='o', alpha=0.7)

        axes[0].set_title(f"Left Image KPs ({len(kp_left_original)})")
        axes[0].axis('off')

        # --- VẼ ẢNH PHẢI ---
        axes[1].imshow(img_right_rgb.astype(np.uint8))

        # Vẽ Keypoint đã Scale
        axes[1].scatter(kp_right_original[:, 0], kp_right_original[:, 1],
                        s=30, c='red', marker='o', alpha=0.7)

        axes[1].set_title(f"Right Image KPs ({len(kp_right_original)})")
        axes[1].axis('off')

        # --- HIỂN THỊ CHUNG ---
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def show_final_panorama(self, image_rgb, title="Final Panorama (Manual Stitching)"):
        """
        Hiển thị ảnh panorama cuối cùng.
        image_rgb: ảnh cuối cùng (phải là RGB)
        """
        plt.figure(figsize=(16, 8))
        plt.imshow(image_rgb.astype(np.uint8))
        plt.title(title)
        plt.axis('off')
        plt.show()
        plt.close()

viz = ImageVisualizer()