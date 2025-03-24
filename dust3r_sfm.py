import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import (get_sorted_image_files, load_images)
from utils.misc import read_params_from_json
from utils.colmap_dataset_utils import save_pointcloud_with_normals

def main(source_path, ckpt_path, device, image_size, schedule, lr, min_conf_thr,
         parmas_file_path = "assets/carla/town10/params"):

    # ---------------- (1) Load model and images ----------------  
    model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    train_img_files = image_files
    # when geometry init, only use train images
    image_files = train_img_files
    images, org_imgs_shape = load_images(image_files, size=image_size)
    sparse_path = Path(source_path) / 'sparse/0'
    os.makedirs(sparse_path, exist_ok=True)
    parmas_files = np.sort(os.listdir(parmas_file_path))
    intrinsics, extrinsics = read_params_from_json(parmas_file_path, parmas_files, if_scale = True,
                                                   old_size=(1920, 1080), new_size=(512, 288))
    
    print(f'>> Making pairs...')
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f'>> Inference...')
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    print(f'>> Global alignment...')
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, fx_and_fy=True)
    scene.preset_pose(extrinsics,[True] * len(extrinsics))
    scene.preset_intrinsics(intrinsics,[True] * len(intrinsics))
    loss = scene.compute_global_alignment(init="known_poses", niter=300, schedule=schedule, lr=lr)

    # Extract scene information
    intrinsics = to_numpy(scene.get_intrinsics())
    pts3d = to_numpy(scene.get_pts3d())
    pts3d = np.array(pts3d)
    depthmaps = to_numpy([param.detach().cpu().numpy() for param in scene.im_depthmaps])
    depthmaps = np.array(depthmaps)
    values = [param.detach().cpu().numpy() for param in scene.im_conf]
    confs = np.array(values)
    imgs = np.array(scene.imgs)
    
    masks = to_numpy([c > min_conf_thr for c in to_numpy(confs)])
    
    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    # Save results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--ckpt_path', type=str,
        default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric', help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--parmas_file_path',type=str, default="assets/carla/town10/params", help='Directory containing cameras parameters.')
    parser.add_argument('--min_conf_thr', type=float, default=2, help='Minimum confidence threshold')

    args = parser.parse_args()
    main(args.source_path, args.ckpt_path, args.device, args.image_size, args.schedule, args.lr, args.min_conf_thr,       
          args.parmas_file_path)
