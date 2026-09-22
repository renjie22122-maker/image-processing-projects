import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
import os
import tkinter as tk
from tkinter import filedialog

class PoissonEditor:
    def __init__(self, source_img, target_img):
        self.source = source_img
        self.target = target_img
        self.mask = None
        self.offset = (0, 0)

    def create_mask_polygon(self):
        if self.source is None: return
        img_copy = self.source.copy()
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
                if len(points) > 1:
                    cv2.line(img_copy, points[-2], points[-1], (0, 255, 0), 1)
                cv2.imshow("Select Mask", img_copy)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(points) > 2:
                    cv2.line(img_copy, points[-1], points[0], (0, 255, 0), 1)
                    pts = np.array(points, dtype=np.int32)
                    self.mask = np.zeros(self.source.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(self.mask, [pts], 255)
                    combined = cv2.addWeighted(img_copy, 0.7, cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
                    cv2.imshow("Select Mask", combined)

        print("[Instructions] Left-click to add points, Right-click to close.")
        cv2.namedWindow("Select Mask")
        cv2.setMouseCallback("Select Mask", mouse_callback)
        cv2.imshow("Select Mask", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_crop_bounds(self, offset_y, offset_x):
        y_indices, x_indices = np.nonzero(self.mask)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        # Bounding box height/width
        h, w = max_y - min_y + 1, max_x - min_x + 1
        
        # Crop Source
        src_crop = self.source[min_y:max_y+1, min_x:max_x+1]
        mask_crop = self.mask[min_y:max_y+1, min_x:max_x+1]
        
        # Crop Target (handle boundaries)
        tgt_start_y, tgt_start_x = min_y + offset_y, min_x + offset_x
        tgt_crop = self.target[tgt_start_y:tgt_start_y+h, tgt_start_x:tgt_start_x+w]
        
        return src_crop, tgt_crop, mask_crop

    def solve_poisson(self, offset_y, offset_x, mode='import', **kwargs):
        if self.mask is None:
            print("Error: No mask.")
            return self.target

        # 1. Optimize: Crop to Bounding Box
        try:
            src_roi, tgt_roi, mask_roi = self.get_crop_bounds(offset_y, offset_x)
        except Exception as e:
            print("Error cropping (likely out of bounds):", e)
            return self.target

        # Normalize mask to 0/1
        mask_bool = (mask_roi > 127)
        H, W = mask_bool.shape
        num_pixels = np.sum(mask_bool)
        
        if num_pixels == 0: return self.target

        # 2. Map 2D coordinates to 1D system index
        id_map = np.full((H, W), -1, dtype=np.int32)
        id_map[mask_bool] = np.arange(num_pixels)

        # 3. Construct Laplacian Matrix A 
        y, x = np.nonzero(mask_bool)
        
        data = np.full(num_pixels, 4.0) # Diagonal is 4
        rows = np.arange(num_pixels)
        cols = np.arange(num_pixels)
        
        neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        all_rows = [rows]
        all_cols = [cols]
        all_data = [data]

        for dy, dx in neighbors_offsets:
            ny, nx = y + dy, x + dx
            valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            
            neighbor_ids = np.full(num_pixels, -1)
            neighbor_ids[valid] = id_map[ny[valid], nx[valid]]
            
            is_variable = neighbor_ids != -1
            
            if np.any(is_variable):
                all_rows.append(rows[is_variable])
                all_cols.append(neighbor_ids[is_variable])
                all_data.append(np.full(np.sum(is_variable), -1.0))

        A = scipy.sparse.coo_matrix(
            (np.concatenate(all_data), (np.concatenate(all_rows), np.concatenate(all_cols))),
            shape=(num_pixels, num_pixels)
        ).tocsr()

        # 4. Construct Guidance Field b 
        result_roi_channels = []
        
        # Pre-process source for Color Change mode
        src_working = src_roi.astype(float)
        if mode == 'color':
            gains = kwargs.get('color_gains', (1.0, 1.0, 1.0)) # B, G, R
            src_working[:, :, 0] *= gains[0]
            src_working[:, :, 1] *= gains[1]
            src_working[:, :, 2] *= gains[2]

        tgt_working = tgt_roi.astype(float)
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        for c in range(3):
            s_ch = src_working[:, :, c]
            t_ch = tgt_working[:, :, c]
            
            if mode == 'zero': # Membrane
                div_v = np.zeros_like(s_ch)
            elif mode == 'import' or mode == 'color':
                div_v = cv2.filter2D(s_ch, -1, lap_kernel)
            elif mode == 'illumination':
                alpha = kwargs.get('illum_alpha', 1.0)
                div_v = alpha * cv2.filter2D(s_ch, -1, lap_kernel)
            elif mode == 'flatten':
                div_v = np.zeros_like(s_ch)
                thresh = kwargs.get('flatten_threshold', 10.0)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    s_neigh = np.roll(s_ch, shift=(-dy, -dx), axis=(0, 1))
                    grad = s_ch - s_neigh
                    grad[np.abs(grad) < thresh] = 0
                    div_v += grad
            elif mode == 'mix':
                div_v = np.zeros_like(s_ch)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    s_neigh = np.roll(s_ch, shift=(-dy, -dx), axis=(0, 1))
                    t_neigh = np.roll(t_ch, shift=(-dy, -dx), axis=(0, 1))
                    grad_src = s_ch - s_neigh
                    grad_tgt = t_ch - t_neigh
                    grad_mix = np.where(np.abs(grad_tgt) > np.abs(grad_src), grad_tgt, grad_src)
                    div_v += grad_mix

            b = div_v[mask_bool].flatten()

            # Boundary
            for dy, dx in neighbors_offsets:
                ny, nx = y + dy, x + dx
                
                valid_idx = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
                
                # Case 1: Inside ROI, Outside Mask
                inside_roi_mask = np.zeros(num_pixels, dtype=bool)
                inside_roi_mask[valid_idx] = ~mask_bool[ny[valid_idx], nx[valid_idx]]
                if np.any(inside_roi_mask):
                    curr_y = y[inside_roi_mask]
                    curr_x = x[inside_roi_mask]
                    b[inside_roi_mask] += t_ch[curr_y + dy, curr_x + dx]
                
                # Case 2: Outside ROI
                outside_roi_mask = ~valid_idx
                if np.any(outside_roi_mask):
                    y_indices, x_indices = np.nonzero(self.mask)
                    min_y, min_x = np.min(y_indices), np.min(x_indices)
                    gy = np.clip(min_y + y[outside_roi_mask] + dy + offset_y, 0, self.target.shape[0]-1)
                    gx = np.clip(min_x + x[outside_roi_mask] + dx + offset_x, 0, self.target.shape[1]-1)
                    b[outside_roi_mask] += self.target[gy, gx, c]

            x_sol = spsolve(A, b)
            res_ch = t_ch.copy()
            res_ch[mask_bool] = np.clip(x_sol, 0, 255)
            result_roi_channels.append(res_ch)

        res_roi = cv2.merge([c.astype(np.uint8) for c in result_roi_channels])
        
        full_result = self.target.copy()
        y_indices, x_indices = np.nonzero(self.mask)
        min_y, min_x = np.min(y_indices), np.min(x_indices)
        tgt_start_y, tgt_start_x = min_y + offset_y, min_x + offset_x
        full_result[tgt_start_y:tgt_start_y+H, tgt_start_x:tgt_start_x+W] = res_roi
        return full_result

    def solve_tiling(self):
        if self.mask is None:
            print("Error: No mask selected for tiling.")
            return None

        # --- 1. Get Bounding Box of the Mask ---
        y_indices, x_indices = np.nonzero(self.mask)
        if len(y_indices) == 0:
            print("Error: Empty mask.")
            return None
            
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        # Crop the source image (Use floating point for calculation)
        patch = self.source[min_y:max_y+1, min_x:max_x+1].astype(float)
        H, W = patch.shape[:2]
        num_pixels = H * W
        
        print(f"Generating Seamless Tile from crop size: {W}x{H}...")

        # --- 2. Construct Periodic Laplacian Matrix A ---
        # Grid indices (0 to N-1)
        idx = np.arange(num_pixels).reshape(H, W)
        
        rows, cols, data = [], [], []
        
        # Diagonal (4)
        rows.append(idx.flatten())
        cols.append(idx.flatten())
        data.append(np.full(num_pixels, 4.0))
        
        # Neighbors with Periodic Wrapping (np.roll handles the wrap-around)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_idx = np.roll(idx, shift=(-dy, -dx), axis=(0, 1))
            
            rows.append(idx.flatten())
            cols.append(neighbor_idx.flatten())
            data.append(np.full(num_pixels, -1.0))
            
        A = scipy.sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(num_pixels, num_pixels)
        ).tocsr()
        
        A[0, :] = 0
        A[0, 0] = 1
        
        # --- 3. Compute Guidance & Solve ---
        res_channels = []
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        
        for c in range(3):
            g = patch[:, :, c]
            
            # Compute Laplacian of the source patch
            b_mat = cv2.filter2D(g, -1, lap_kernel, borderType=cv2.BORDER_REPLICATE)
            b = b_mat.flatten()
            
            # Constraint for the fixed pixel
            b[0] = g[0, 0]
            
            # Solve
            x = spsolve(A, b)
            res_channels.append(x.reshape(H, W))
            
        merged = cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in res_channels])
        return merged

# --- Utilities ---
def select_file(title):
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title=title)
    root.destroy()
    return p

# --- Main ---
if __name__ == "__main__":
    RESULTS_DIR = "Results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=== Poisson Editor ===")
    
    # --- TASK 1: Membrane ---
    print("\n[Task 1] Membrane Interpolation")
    path = select_file("Select Grayscale Image")
    if path:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
        ed = PoissonEditor(img_bgr, img_bgr)
        ed.create_mask_polygon()
        if ed.mask is not None:
            res = ed.solve_poisson(0, 0, mode='zero')
            cv2.imwrite(f"{RESULTS_DIR}/task1_membrane.png", cv2.cvtColor(res, cv2.COLOR_BGR2GRAY))
            print("Saved Task 1.")

    # --- TASK 2/3: Cloning ---
    print("\n[Task 2/3] Seamless Cloning")
    src_p = select_file("Source"); tgt_p = select_file("Target")
    if src_p and tgt_p:
        ed = PoissonEditor(cv2.imread(src_p), cv2.imread(tgt_p))
        ed.create_mask_polygon()
        if ed.mask is not None:
            # Task 2a Import
            res = ed.solve_poisson(50, 50, mode='import')
            cv2.imwrite(f"{RESULTS_DIR}/task2a_import.png", res)
            
            # Task 2b Mix
            res = ed.solve_poisson(50, 50, mode='mix')
            cv2.imwrite(f"{RESULTS_DIR}/task2b_mix.png", res)
            print("Saved Task 2.")

    # --- TASK 4: The 4 Sub-Tasks ---
    print("\n[Task 4] Implementation of All Effects")
    path = select_file("Select Image for Task 4")
    
    if path:
        img = cv2.imread(path)
        ed = PoissonEditor(img, img)
        
        # 4.1 Texture Flattening
        print("--- 4.1 Texture Flattening ---")
        ed.create_mask_polygon()
        if ed.mask is not None:
            res = ed.solve_poisson(0, 0, mode='flatten', flatten_threshold=20.0)
            cv2.imwrite(f"{RESULTS_DIR}/task4_flatten.png", res)
            cv2.imshow("Flattened", res)
            cv2.waitKey(0)
        
        # 4.2 Local Illumination
        print("--- 4.2 Local Illumination Change ---")
        if ed.mask is not None:
            res = ed.solve_poisson(0, 0, mode='illumination', illum_alpha=2.0)
            cv2.imwrite(f"{RESULTS_DIR}/task4_illumination.png", res)
            cv2.imshow("Illumination Boost", res)
            cv2.waitKey(0)

        # 4.3 Local Color Change
        print("--- 4.3 Local Color Change ---")
        if ed.mask is not None:
            res = ed.solve_poisson(0, 0, mode='color', color_gains=(0.2, 2.0, 0.2))
            cv2.imwrite(f"{RESULTS_DIR}/task4_color.png", res)
            cv2.imshow("Color Change", res)
            cv2.waitKey(0)
            
        # 4.4 Seamless Tiling
        print("--- 4.4 Seamless Tiling ---")
        print("Draw a MASK around the texture patch you want to make seamless.")
        ed.mask = None # Reset mask
        ed.create_mask_polygon()
        
        if ed.mask is not None:
            res_tile = ed.solve_tiling()
            
            if res_tile is not None:
                cv2.imwrite(f"{RESULTS_DIR}/task4_tiling.png", res_tile)
                
                # Show a 2x2 grid to prove it works
                h, w = res_tile.shape[:2]
                grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                grid[0:h, 0:w] = res_tile
                grid[0:h, w:] = res_tile
                grid[h:, 0:w] = res_tile
                grid[h:, w:] = res_tile
                cv2.imshow("2x2 Tiling Check", grid)
                cv2.waitKey(0)

    cv2.destroyAllWindows()
