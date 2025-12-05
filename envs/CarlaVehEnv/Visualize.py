import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from PIL import Image

def pointcloud_to_bev(points, res=0.1, side_range=(-10, 10), fwd_range=(0, 20)):
    x_points = points[:, 0]
    y_points = points[:, 1]

    mask = (x_points > fwd_range[0]) & (x_points < fwd_range[1]) & \
           (y_points > side_range[0]) & (y_points < side_range[1])
    x_points = x_points[mask]
    y_points = y_points[mask]

    x_img = ((x_points - fwd_range[0]) / res).astype(np.int32)
    y_img = ((y_points - side_range[0]) / res).astype(np.int32)

    x_max = int((fwd_range[1] - fwd_range[0]) / res)
    y_max = int((side_range[1] - side_range[0]) / res)

    bev_image = np.zeros((y_max, x_max), dtype=np.uint8)
    bev_image[y_img, x_img] = 255

    return cv2.cvtColor(bev_image, cv2.COLOR_GRAY2RGB)

def load_visual(filepath, use_colormap=False):
    if filepath.endswith(('.png', '.jpg', '.jpeg')):
        # 加载图像文件为 RGB 格式
        return np.array(Image.open(filepath).convert('RGB'))

    elif filepath.endswith('.npz'):
        # 加载 .npz 文件并提取图像数组
        npz = np.load(filepath)
        if 'image' in npz:
            arr = npz['image']
        elif 'arr_0' in npz:
            arr = npz['arr_0']
        else:
            raise ValueError(f"{filepath} is missing 'image' or 'arr_0' field.")

        # 归一化并转换为热力图
        arr = (arr - np.min(arr)) / (np.ptp(arr) + 1e-6)
        arr_uint8 = (arr * 255).astype(np.uint8)
        if use_colormap:
            img_color = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_JET)
            return cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        else:
            return np.stack([arr_uint8]*3, axis=-1)

    elif filepath.endswith('.pcd.bin'):
        pc_raw = np.fromfile(filepath, dtype=np.float32)
        size = pc_raw.size

        if size % 5 == 0:
            pc = pc_raw.reshape(-1, 5)
        elif size % 4 == 0:
            pc = pc_raw.reshape(-1, 4)
        elif size % 3 == 0:
            pc = pc_raw.reshape(-1, 3)
            # 若缺失 intensity，添加全 0 的占位列
            pc = np.concatenate([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)], axis=1)
        else:
            raise ValueError(f"{filepath} has invalid size {size}, cannot reshape into (N, 3/4/5).")

        # 使用前三个维度 (x, y, z) 可视化
        bev = pointcloud_to_bev(pc[:, :3])
        return bev

    else:
        raise ValueError(f"Unsupported file format: {filepath}")



def visualize_step(step_idx, files, base_path, fig, axs):
    fig.suptitle(f"Step {step_idx:03d}")

    axs = axs.flatten()

    for obj_idx in range(min(len(files), len(axs))):
        ax = axs[obj_idx]
        ax.clear()  # 清空旧内容

        fname = files[obj_idx]
        if fname is None:
            ax.set_title(f"Obj {obj_idx} (None)")
            ax.axis('off')
            continue

        fpath = os.path.join(base_path, fname)
        if os.path.exists(fpath):
            try:
                img = load_visual(fpath)
                ax.imshow(img)
                ax.set_title(f"Obj {obj_idx}")
            except Exception as e:
                #print(f"[WARN] Error loading {fpath}: {e}")
                ax.set_title(f"Obj {obj_idx} (Error)")
        else:
            ax.set_title(f"Obj {obj_idx} (Missing)")
        ax.axis('off')

    for i in range(len(files), len(axs)):
        axs[i].clear()
        axs[i].axis('off')

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.1)
