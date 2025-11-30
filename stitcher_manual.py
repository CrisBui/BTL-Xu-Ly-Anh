import numpy as np
import math
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist
from ImageVisualizer import viz

class PanoramaStitcher:
    def __init__(self):
        self.process_width = 600 
        self.sift_sigma = 1.6
        self.ratio_test = 0.75

    # --- UTILS ---
    def load_image(self, path):
        """ Load image using PIL and convert to NumPy array """
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def save_image(self, img_array, path):
        """ Save NumPy array to disk using PIL """
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(path)

    def resize_image(self, img, width):
        """ Resize using PIL """
        h, w = img.shape[:2]
        scale = width / w
        new_h = int(h * scale)
        pil_img = Image.fromarray(img)
        resized = pil_img.resize((int(width), new_h), Image.BILINEAR)
        return np.array(resized), scale

    def to_gray(self, img):
        """ RGB to Grayscale manually (luminosity formula) """
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    # --- MANUAL SIFT IMPLEMENTATION (Simplified) ---
    def compute_sift(self, img_gray):
        """ 
        A simplified SIFT:
        1. Difference of Gaussians (DoG) for keypoints
        2. Gradient Histograms for descriptors
        """
        # 1. DoG Detector
        s = 1.6
        k = 2 ** 0.5
        # Blur images
        g1 = ndimage.gaussian_filter(img_gray, s)
        g2 = ndimage.gaussian_filter(img_gray, s * k)
        g3 = ndimage.gaussian_filter(img_gray, s * k * k)
        
        # Difference
        dog1 = g2 - g1
        dog2 = g3 - g2
        
        # Find local extrema (peaks in 3x3 region) using max filter
        local_max = ndimage.maximum_filter(dog1, size=3) == dog1
        threshold = 0.03 * (dog1.max() - dog1.min()) 
        keypoints = np.argwhere(local_max & (np.abs(dog1) > threshold))
        
        # Limit keypoints for speed (Top 500 strongest)
        if len(keypoints) > 500:
            strengths = dog1[keypoints[:,0], keypoints[:,1]]
            indices = np.argsort(strengths)[-500:]
            keypoints = keypoints[indices]

        # 2. Descriptors (Simplified 128-dim vector)
        # Calculate gradients
        hy, hx = np.gradient(g1)
        magnitude = np.sqrt(hx**2 + hy**2)
        orientation = np.arctan2(hy, hx) * (180 / np.pi) % 360

        descriptors = []
        final_kps = []

        # Create 4x4 grid descriptors
        # We extract a 16x16 patch around keypoint, divide into 4x4 cells
        bin_size = 8
        for kp in keypoints:
            y, x = kp
            # Boundary check
            if y < 8 or y > img_gray.shape[0]-9 or x < 8 or x > img_gray.shape[1]-9:
                continue
                
            patch_mag = magnitude[y-8:y+8, x-8:x+8]
            patch_ori = orientation[y-8:y+8, x-8:x+8]
            
            vector = []
            # 4x4 grid cells (each cell is 4x4 pixels)
            for i in range(0, 16, 4):
                for j in range(0, 16, 4):
                    cell_mag = patch_mag[i:i+4, j:j+4]
                    cell_ori = patch_ori[i:i+4, j:j+4]
                    
                    # 8-bin histogram for this cell
                    hist, _ = np.histogram(cell_ori, bins=8, range=(0, 360), weights=cell_mag)
                    vector.extend(hist)
            
            # Normalize vector (L2 norm) to handle illumination changes
            vector = np.array(vector)
            norm = np.linalg.norm(vector)
            if norm > 0: vector /= norm
            
            descriptors.append(vector)
            final_kps.append([x, y]) # Store as X, Y

        return np.array(final_kps), np.array(descriptors)


    # --- MANUAL MATCHING ---
    def match_features(self, des1, des2):
        """ Brute Force Matching with Lowe's Ratio Test """
        if len(des1) == 0 or len(des2) == 0: return []
        
        # Compute euclidean distance between all pairs
        dists = cdist(des1, des2, 'euclidean')
        
        matches = []
        for i in range(len(dists)):
            # Sort distances for this query descriptor
            sorted_indices = np.argsort(dists[i])
            best_idx = sorted_indices[0]
            second_idx = sorted_indices[1]
            
            # Lowe's Ratio Test
            if dists[i][best_idx] < self.ratio_test * dists[i][second_idx]:
                matches.append((i, best_idx)) # (queryIdx, trainIdx)
        
        return matches

    # --- MANUAL HOMOGRAPHY (DLT) ---
    def find_homography_ransac(self, src_pts, dst_pts, threshold=5.0):
        """ 
        RANSAC loop to find best Homography using DLT.
        src_pts: points in image to be warped
        dst_pts: points in destination image
        """
        max_inliers = 0
        best_H = None
        
        n_points = len(src_pts)
        if n_points < 4: return None
        
        # Add homogenous coordinate (1)
        src_h = np.hstack([src_pts, np.ones((n_points, 1))])
        dst_h = np.hstack([dst_pts, np.ones((n_points, 1))])

        iterations = 500 # Limit iterations for pure python speed
        
        for _ in range(iterations):
            # 1. Randomly select 4 points
            indices = np.random.choice(n_points, 4, replace=False)
            s_sample = src_pts[indices]
            d_sample = dst_pts[indices]
            
            # 2. Compute H using DLT (Direct Linear Transform)
            # Construct matrix A
            A = []
            for i in range(4):
                x, y = s_sample[i]
                u, v = d_sample[i]
                A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
                A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
            A = np.array(A)
            
            # Solve using SVD
            try:
                U, S, Vh = np.linalg.svd(A)
                H = Vh[-1].reshape(3, 3) # The last row of Vh is the solution
            except np.linalg.LinAlgError:
                continue

            # Normalize H (h33 = 1)
            if H[2, 2] == 0: continue
            H = H / H[2, 2]
            
            # 3. Count Inliers
            # Project all points using H
            projected = np.dot(src_h, H.T)
            # Normalize homogeneous coordinates
            with np.errstate(divide='ignore', invalid='ignore'):
                projected = projected[:, :2] / projected[:, 2:]
            
            # Euclidean distance
            dists = np.linalg.norm(dst_pts - projected, axis=1)
            inliers = np.sum(dists < threshold)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H
                
        return best_H

    # --- MANUAL WARPING ---
    def warp_perspective_manual(self, img, H, output_shape):
        """
        Inverse Mapping Warping using scipy map_coordinates.
        This avoids writing a slow Python loop over pixels.
        """
        w, h = output_shape
        # Create grid of destination coordinates (x, y)
        x_range = np.arange(w)
        y_range = np.arange(h)
        xv, yv = np.meshgrid(x_range, y_range)
        
        # Flatten to list of points
        ones = np.ones_like(xv)
        coords = np.stack([xv, yv, ones])
        coords_flat = coords.reshape(3, -1)
        
        # Inverse warp: Map destination pixels back to source image
        try:
            H_inv = np.linalg.inv(H)
        except:
            return np.zeros((h, w, 3), dtype=np.uint8)

        src_coords_hom = np.dot(H_inv, coords_flat)
        
        # Normalize homogeneous
        src_coords_hom /= (src_coords_hom[2] + 1e-10) 
        src_x = src_coords_hom[0].reshape(h, w)
        src_y = src_coords_hom[1].reshape(h, w)
        
        # Map colors using interpolation
        # We process each channel (R, G, B) separately
        warped = np.zeros((h, w, 3))
        for i in range(3):
            warped[:, :, i] = ndimage.map_coordinates(
                img[:, :, i], [src_y, src_x], order=1, mode='constant', cval=0
            )
            
        return warped.astype(np.uint8)

    def crop_borders(self, img):
        # Sử dụng ngưỡng nhỏ (ví dụ: 5 trên thang 0-255)
        GRAY_THRESHOLD = 1

        gray = self.to_gray(img)

        # Tìm các hàng và cột CÓ pixel màu sáng hơn ngưỡng
        rows = np.any(gray > GRAY_THRESHOLD, axis=1)
        cols = np.any(gray > GRAY_THRESHOLD, axis=0)

        # Kiểm tra nếu không có pixel nào sáng hơn ngưỡng (Tránh lỗi np.where)
        if not np.any(rows) or not np.any(cols):
            # Trả lại ảnh nếu nó gần như hoàn toàn là màu đen/xám tối
            return img

            # Lấy chỉ số đầu tiên (rmin/cmin) và cuối cùng (rmax/cmax) của các hàng/cột TRUE
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Cắt ảnh
        return img[rmin:rmax + 1, cmin:cmax + 1]

    # --- PIPELINE ---
    def stitch_pair(self, img_left, img_right):
        # 1. Resize for Speed (Critical for pure python SIFT)
        img_l_small, scale_l = self.resize_image(img_left, self.process_width)
        img_r_small, scale_r = self.resize_image(img_right, self.process_width)
        
        # 2. Manual SIFT
        kpr, des1 = self.compute_sift(self.to_gray(img_r_small))
        kpl, des2 = self.compute_sift(self.to_gray(img_l_small))

        if len(kpr) == 0 or len(kpl) == 0: return img_left

        # 3. Match
        matches = self.match_features(des1, des2)
        if len(matches) < 4: return img_left

        matched_img = viz.draw_keypoint_and_matches_unified(
            img_left_rgb=img_left,           # Ảnh gốc
            kp_left_small=kpl,               # Tọa độ NHỎ
            scale_left=scale_l,              # Hệ số scale
            img_right_rgb=img_right,         # Ảnh gốc
            kp_right_small=kpr,              # Tọa độ NHỎ
            scale_right=scale_r,             # Hệ số scale
            matches=matches,
            title="SIFT Unified Visualization (Before RANSAC)"
        )
        
        # 4. Homography
        src_pts = np.float32([kpr[m[0]] for m in matches]) * (1/scale_r)
        dst_pts = np.float32([kpl[m[1]] for m in matches]) * (1/scale_l)
        
        H = self.find_homography_ransac(src_pts, dst_pts)
        if H is None: return img_left

        ransac_matched_img = viz.draw_matches_after_ransac(
            img_left_rgb=img_left, kp_left_small=kpl, scale_left=scale_l,
            img_right_rgb=img_right, kp_right_small=kpr, scale_right=scale_r,
            matches=matches,
            H=H,  # Truyền ma trận Homography đã tìm được
            threshold=5.0,  # Sử dụng cùng ngưỡng RANSAC
            title="SIFT Inliers Visualization (After RANSAC)"
        )

        # 5. Calculate Canvas
        h1, w1 = img_right.shape[:2]
        h2, w2 = img_left.shape[:2]
        
        # Transform corners to find canvas size
        corners = np.float32([[0, 0, 1], [0, h1, 1], [w1, h1, 1], [w1, 0, 1]]).T
        warped_corners = np.dot(H, corners)
        warped_corners /= warped_corners[2]
        
        all_x = np.concatenate(([0, w2], warped_corners[0]))
        all_y = np.concatenate(([0, h2], warped_corners[1]))
        
        xmin, xmax = int(np.min(all_x)), int(np.max(all_x))
        ymin, ymax = int(np.min(all_y)), int(np.max(all_y))
        
        translation = [-xmin, -ymin]
        H_trans = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
        
        # 6. Warp
        full_H = np.dot(H_trans, H)
        output_dims = (xmax - xmin, ymax - ymin)
        
        # Warp right image
        warped_right = self.warp_perspective_manual(img_right, full_H, output_dims)
        
        # Place left image
        y_off, x_off = translation[1], translation[0]
        
        # Manual overlay
        h_res, w_res = warped_right.shape[:2]
        # Ensure bounds
        y_end = min(y_off+h2, h_res)
        x_end = min(x_off+w2, w_res)
        
        mask_l = np.any(img_left > 0, axis=2)
        
        # Region of interest in canvas
        roi = warped_right[y_off:y_end, x_off:x_end]
        
        # Where left image is valid, replace canvas
        roi_mask = mask_l[:y_end-y_off, :x_end-x_off]
        roi[roi_mask] = img_left[:y_end-y_off, :x_end-x_off][roi_mask]
        
        warped_right[y_off:y_end, x_off:x_end] = roi
        
        return warped_right.astype(np.uint8), matched_img.astype(np.uint8), ransac_matched_img.astype(np.uint8)

    def stitch(self, image_paths):
        images = [self.load_image(p) for p in image_paths]
        if len(images) < 2: return None
        
        result = images[0]
        matched_pairs = []
        ransac_matched_pairs = []
        for i in range(1, len(images)):
            result, matched, ransac_matched = self.stitch_pair(result, images[i])
            matched_pairs.append(matched)
            ransac_matched_pairs.append(ransac_matched)

        result = self.crop_borders(result)
        return result, matched_pairs, ransac_matched_pairs