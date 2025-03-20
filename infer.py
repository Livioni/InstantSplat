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
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import ( save_points3D, save_time, init_filestructure, get_sorted_image_files, split_train_test, load_images)
from misc import read_params_from_json

def main(source_path, model_path, model_name, device, image_size, schedule, lr, niter, 
         min_conf_thr, llffhold, n_views, co_vis_dsp, depth_thre, infer_video=False):

    # ---------------- (1) Load model and images ----------------  
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    if model_name == 'MASt3R':
        model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    elif model_name == 'DUSt3R':
        model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    else:
        raise ValueError(f'Invalid model: {model}')
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    if infer_video:
        train_img_files = image_files
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
    
    # when geometry init, only use train images
    image_files = train_img_files
    images, org_imgs_shape = load_images(image_files, size=image_size)
    parmas_file_path = "assets/carla/town10/params"
    parmas_files = np.sort(os.listdir(parmas_file_path))
    intrinsics, extrinsics = read_params_from_json(parmas_file_path, parmas_files, old_size=(1920, 1080), new_size=(512, 288))
    
    start_time = time()
    print(f'>> Making pairs...')
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f'>> Inference...')
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    print(f'>> Global alignment...')
    if model_name == 'MASt3R':
        lr1=0.07
        niter1=500
        lr2=0.014
        niter2=300
        matching_conf_thr=5.0
        optim_level='refine+depth'
        shared_intrinsics = True
        cache_dir = os.path.join(save_path, 'cache')
        scene = sparse_global_alignment(image_files, pairs, cache_dir,
                                        model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                        opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                        matching_conf_thr=matching_conf_thr)
    else:
        scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, fx_and_fy=True)
        scene.preset_pose(extrinsics,[True] * len(extrinsics))
        scene.preset_intrinsics(intrinsics,[True] * len(intrinsics))
        loss = scene.compute_global_alignment(init="known_poses", niter=300, schedule=schedule, lr=lr)

    # Extract scene information
    if model_name == 'DUSt3R':
        intrinsics = to_numpy(scene.get_intrinsics())
        focals = to_numpy(scene.get_focals())
        imgs = np.array(scene.imgs)
        pts3d = to_numpy(scene.get_pts3d())
        depthmaps = to_numpy([param.detach().cpu().numpy() for param in scene.im_depthmaps])
        confs = [param.detach().cpu().numpy() for param in scene.im_conf]
    elif model_name == 'MASt3R':
        focals = to_numpy(scene.get_focals())
        imgs = np.array(scene.imgs)
        pts3d, depthmaps, confs = scene.get_dense_pts3d()
        pts3d = to_numpy([pts.detach().cpu().numpy() for pts in pts3d])
        depthmaps = to_numpy([depth.detach().cpu().numpy() for depth in depthmaps])
        confs = to_numpy([conf.detach().cpu().numpy() for conf in confs])

    else:
        raise ValueError(f'Invalid model: {model_name}')
    
    pts3d = np.array(pts3d)
    depthmaps = np.array(depthmaps)
    confs = np.array(confs)
    # Save results
    focals = np.repeat(focals[0], n_views)
    print(f'>> Saving results...')
    end_time = time()
    save_time(model_path, '[1] init_geo', end_time - start_time)
    pts_num = save_points3D(sparse_0_path, imgs, pts3d, confs.reshape(pts3d.shape[0], -1), use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=model_path, depth_threshold=depth_thre)
    print(f'[INFO] MASt3R Reconstruction is successfully converted to COLMAP files in: {str(sparse_0_path)}')
    print(f'[INFO] Number of points: {pts3d.reshape(-1, 3).shape[0]}')    
    print(f'[INFO] Number of points after downsampling: {pts_num}')
    
    scene.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--model', type=str, default='MASt3R', help='model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--min_conf_thr', type=float, default=5, help='Minimum confidence threshold')
    parser.add_argument('--llffhold', type=int, default=8, help='')
    parser.add_argument('--n_views', type=int, default=3, help='')
    parser.add_argument('--co_vis_dsp', action="store_true")
    parser.add_argument('--depth_thre', type=float, default=0.01, help='Depth threshold')
    parser.add_argument('--infer_video', action="store_true")

    args = parser.parse_args()
    main(args.source_path, args.model_path, args.model, args.device, args.image_size, args.schedule, args.lr, args.niter,         
          args.min_conf_thr, args.llffhold, args.n_views, args.co_vis_dsp, args.depth_thre, args.infer_video)
