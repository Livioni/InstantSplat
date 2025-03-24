from scipy.spatial.transform import Rotation as R
from copy import copy
import numpy as np
import cv2, os, json
import open3d as o3d

def scale_intrinsics(K, old_size, new_size):
    new_width, new_height = new_size
    old_width, old_height = old_size
    scale_x = new_width / old_width
    scale_y = new_height / old_height

    K_scaled = copy(K) 
    K_scaled[0,0] *= scale_x
    K_scaled[0,2] *= scale_x
    K_scaled[1,1] *= scale_y
    K_scaled[1,2] *= scale_y
    
    return K_scaled


def pose_unreal2opencv(c2w_mat):
    translation = c2w_mat[:3, 3]
    rot = R.from_matrix(c2w_mat[:3, :3])
    rot_vec = rot.as_rotvec()

    rot_vec_new = rot_vec[[1, 2, 0]]
    rot_vec_new[0] *= -1
    rot_vec_new[2] *= -1

    rot = R.from_rotvec(rot_vec_new)
    
    translation_new = translation[[1, 2, 0]]
    translation_new[1] *= -1

    c2w_mat = np.eye(4)
    c2w_mat[:3, :3] = rot.as_matrix()
    c2w_mat[:3, 3] = translation_new

    rot = np.eye(4)
    rot[1,1]=-1
    rot[2, 2] = -1
    c2w_mat =  rot @ c2w_mat
    return c2w_mat


def depth_to_pointcloud(
    depth_map: np.ndarray,
    color_image: np.ndarray,
    fx=525.0,  # 焦距（示例值）
    fy=525.0,  # 焦距（示例值）
    cx=None,   # 主点 x 坐标
    cy=None    # 主点 y 坐标
):
    """
    将深度图和对应的彩色图转换为点云（Open3D格式）
    depth_map: HxW, float 类型
    color_image: HxW, BGR 通道
    fx, fy, cx, cy: 相机内参
    """
    h, w = depth_map.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # 确保颜色图与深度图尺寸一致
    if color_image.shape[0] != h or color_image.shape[1] != w:
        raise ValueError("Color image and depth map size do not match!")
    
    # 转成 RGB 通道再归一化到 [0,1]，以满足 Open3D
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image = color_image.astype(np.float32) / 255.0
    
    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            Z = depth_map[v, u]
            # 过滤掉无效或距离为 0 的点
            if Z <= 0:
                continue
            
            # 根据 pinhole 相机模型反投影到 3D
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            points.append((X, Y, Z))
            colors.append(color_image[v, u])

    # 使用 open3d 构建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float32))
    return pcd

def read_params_from_json(root_path, files, if_scale=False, old_size=(1920,1080), new_size=(512, 288)):
    intrinsics = []
    extrinsics = []
    for parmas_file in files:
        file_path = os.path.join(root_path, parmas_file)
        # 读取 JSON
        with open(file_path, "r") as f:
            data = json.load(f)
        K = np.around(np.array(data["intrinsic"]["K"]),decimals=4)
        T = np.around(np.array(data["extrinsic"]["T"]),decimals=4)
        if if_scale:
            K = scale_intrinsics(K, old_size, new_size)
        T = pose_unreal2opencv(T)
        intrinsics.append(K)
        extrinsics.append(T)
    return intrinsics, extrinsics