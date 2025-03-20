import viser
import numpy as np
import time
import argparse

def visualize_single_npy_cloud(
    points_path="assets/carla/town10/sparse_8/0/points3D_all.npy",
    color_path="assets/carla/town10/sparse_8/0/pointsColor_all.npy",
    conf_path="assets/carla/town10/sparse_8/0/confidence.npy",
    port=7869,
    initial_point_size=0.02,
):
    """
    用 viser 可视化一个单点云 (N,3) ，并配合置信度 (N,) 实时筛选。

    参数：
    - points_path: 3D坐标的 .npy
    - color_path: 颜色的 .npy (与 points 顺序一致)
    - conf_path: 置信度的 .npy (与 points 顺序一致)
    - port: viser 启动的端口
    - initial_point_size: 初始点大小
    """

    print(f"Loading points from {points_path}")
    points = np.load(points_path).astype(np.float32)  # shape (N,3)
    points = points.reshape(-1, 3)  # 确保是二维的
    print(f"Loading colors from {color_path}")
    
    colors = np.load(color_path).astype(np.float32)   # shape (N,3)
    colors = colors.reshape(-1, 3)  # 确保是二维的
    # 若颜色范围在 [0,255]，需归一化到 [0,1]
    if colors.max() > 1.0:
        colors /= 255.0

    print(f"Loading confidence from {conf_path}")
    conf = np.load(conf_path).astype(np.float32)      # shape (N,)
    conf = conf.reshape(-1)  # 确保是一
    assert len(conf) == len(points), (
        f"confidence.npy({len(conf)}) 与 points3D_all.npy({len(points)}) 行数不匹配！"
    )

    # 归一化后的置信度可用于着色
    conf_min, conf_max = conf.min(), conf.max()
    conf_range = conf_max - conf_min + 1e-8
    conf_norm = (conf - conf_min) / conf_range  # in [0, 1]

    # 生成置信度colormap，比如 turbo
    from matplotlib import cm
    cm_turbo = cm.get_cmap("turbo")
    conf_colors = cm_turbo(conf_norm)[:, :3]  # shape (N,3), float in [0,1]

    # 启动 viser 服务器
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.gui.set_panel_label("Show Controls")

    # 添加一个点云节点
    pointcloud_node = server.scene.add_point_cloud(
        name="/my_pointcloud",
        points=points,      # 初始显示全部点
        colors=colors,      # 初始用原始颜色
        point_size=initial_point_size,
        point_shape="rounded",
        visible=True,
    )

    # ---------- GUI 控件 ----------

    # 1) 最小置信度滑块
    min_conf_slider = server.gui.add_slider(
        label="Confidence Threshold",
        min=float(conf_min),
        max=float(conf_max),
        step=(conf_range / 100.0),
        initial_value=float(conf_min),
    )

    # 2) 复选框: 是否用置信度着色
    color_by_conf_checkbox = server.gui.add_checkbox("Color by Confidence", False)

    # 3) 点大小
    gui_point_size = server.gui.add_slider("Point Size", 
                                           min=0.001, 
                                           max=0.1, 
                                           step=1e-4, 
                                           initial_value=initial_point_size)


    # 回调函数：当滑块或复选框变动，动态更新显示
    @min_conf_slider.on_update
    def update_conf_filter(event: viser.GuiEvent) -> None:
        threshold = min_conf_slider.value
        # 筛选出 >= 阈值的点
        mask = conf >= threshold

        # 是否用置信度着色
        use_conf_color = color_by_conf_checkbox.value
        updated_colors = conf_colors[mask] if use_conf_color else colors[mask]

        pointcloud_node.points = points[mask]
        pointcloud_node.colors = updated_colors
        print(f"[Slider] threshold={threshold:.3f}, showing {mask.sum()}/{len(conf)} points")

    @color_by_conf_checkbox.on_update
    def update_color_mode(event: viser.GuiEvent) -> None:
        # 不改变阈值，只改颜色模式
        threshold = min_conf_slider.value
        mask = conf >= threshold
        if color_by_conf_checkbox.value:
            # 用置信度着色
            pointcloud_node.colors = conf_colors[mask]
        else:
            # 用原图颜色
            pointcloud_node.colors = colors[mask]

        print(f"[Checkbox] color by confidence={color_by_conf_checkbox.value}, still showing {mask.sum()} points")

    @gui_point_size.on_update
    def _(_):
        with server.atomic():
            pointcloud_node.point_size = gui_point_size.value
        print(f"[Slider] updated point size to {gui_point_size.value:g}")
        server.flush()
        

    print(f"\nViser server started at: http://127.0.0.1:{port}")
    print("Use the slider 'Confidence Threshold' on the left to filter points.")
    print("Check 'Color by Confidence' to visualize confidence as a colormap.")
    print("Press Ctrl+C to quit.")
    
    # 阻塞主进程，让服务器一直运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--points_path', type=str,  default="assets/carla/town10/sparse_8/0/points3D_all.npy",  help='Directory to save the points')
    parser.add_argument('--color_path', type=str,  default="assets/carla/town10/sparse_8/0/pointsColor_all.npy", help='Directory to save the colors')
    parser.add_argument('--conf_path', type=str,  default="assets/carla/town10/sparse_8/0/confidence.npy" ,help='Directory to save the confidences')

    args = parser.parse_args()

    visualize_single_npy_cloud(points_path=args.points_path,
                               color_path=args.color_path,
                               conf_path=args.conf_path)