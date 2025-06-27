# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/evaluate_iterative_single_CALIB.py

import csv
import random
import open3d as o3

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm
import time
import shutil

from models.LCCNet import LCCNet
from DatasetLidarCamera import DatasetLidarCameraKittiOdometry

from quaternion_distances import quaternion_distance
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)


plt.rcParams['axes.unicode_minus'] = False
font_EN = {'family': 'sans-serif', 'weight': 'normal', 'size': 16}
font_CN = {'family': 'AR PL UMing CN', 'weight': 'normal', 'size': 16}
plt_size = 10.5

ex = Experiment("LCCNet-evaluate-iterative", save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'kitti/odom'
    data_folder = './datasets/own_data_situation_split'
    #data_folder = './datasets/odometry_color_short'
    img_shape  = (2128, 2600)  # padded image resolution (H, W)  # KITTI: (384, 1280)  # Own: (2128, 2600)
    #img_shape  = (384, 1280)  # padded image resolution (H, W)  # KITTI: (384, 1280)  # Own: (2128, 2600)
    input_size = (256, 512)  # network input resolution (H, W)  # KITTI: (256, 512)  # Own: (256, 512)
    test_sequence = 0
    use_prev_output = False
    max_t = 1.5
    max_r = 20.0
    occlusion_kernel = 5  # nowhere used
    occlusion_threshold = 3.0  # nowhere used
    network = 'Res_f1'
    norm = 'bn'
    show = False
    use_reflectance = False
    weight = None  # List of weights' path, for iterative refinement
    save_name = True
    rot_transl_separated = False  # Set to True only if you use two networks, the first for rotation and the second for translation
    random_initial_pose = False
    save_log = False
    dropout = 0.0
    max_depth = 80.
    iterative_method = 'multi_range' # ['multi_range', 'single_range', 'single']
    output = './output'
    save_image = True
    outlier_filter = False
    outlier_filter_th = 10
    out_fig_lg = 'EN' # [EN, CN]

'''
weights = [
    './pretrained/kitti_iter1.pth',
    './pretrained/kitti_iter2.pth',
    './pretrained/kitti_iter3.pth',
    './pretrained/kitti_iter4.pth',
    './pretrained/kitti_iter5.pth'
]
'''


weights = [
    './checkpoints_fixed_training/kitti/odom/val_seq_00/models/checkpoint_r20.00_t1.50_e1_2.335.pth'
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).to(device)
    depth_img = depth_img.permute(2, 0, 1)
    pc_valid = pc_rotated.T[mask]

    return depth_img, pcl_uv, pc_valid


def project_pointcloud(pc_xyz, extrinsic, cam_intrinsic):
    """Project 3D point cloud to 2D image plane using given extrinsic and intrinsics."""
    pc_homo = np.hstack((pc_xyz, np.ones((pc_xyz.shape[0], 1)))).T  # (4, N)
    pc_transformed = np.dot(extrinsic, pc_homo)
    pc_transformed = pc_transformed[:3, :].T
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_transformed, cam_intrinsic)
    return pcl_uv, pcl_z


@ex.automain
def main(_config, seed):
    global EPOCH, weights
    if _config['weight'] is not None:
        weights = _config['weight']

    if _config['iterative_method'] == 'single':
        weights = [weights[0]]

    dataset_class = DatasetLidarCameraKittiOdometry
    # dataset_class = DatasetTest

    img_shape   = tuple(_config['img_shape'])
    input_size  = tuple(_config['input_size'])

    # split = 'test'
    if _config['random_initial_pose']:
        split = 'test_random'

    if _config['test_sequence'] is None:
        raise TypeError('test_sequences cannot be None')
    else:
        # compute a formatted, two-digit test_sequence without mutating Sacred's config
        if isinstance(_config['test_sequence'], int):
            test_sequence = f"{_config['test_sequence']:02d}"
        else:
            test_sequence = _config['test_sequence']

        dataset_val = dataset_class(
            _config['data_folder'],
            max_r=_config['max_r'],
            max_t=_config['max_t'],
            split='test',
            use_reflectance=_config['use_reflectance'],
            val_sequence=test_sequence,
            device=device
        )

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x):
        return _init_fn(x, seed)

    num_worker = 6
    batch_size = 1

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    print("Test image length:", len(TestImgLoader))

    models = [] # iterative model
    for i in range(len(weights)):
        # network choice and settings
        if _config['network'].startswith('Res'):
            feat = 1
            md = 4
            split = _config['network'].split('_')
            for item in split[1:]:
                if item.startswith('f'):
                    feat = int(item[-1])
                elif item.startswith('md'):
                    md = int(item[2:])
            assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
            assert 0 < md, "md must be positive"
            model = LCCNet(input_size, use_feat_from=feat, md=md,
                             use_reflectance=_config['use_reflectance'], dropout=_config['dropout'])
        else:
            raise TypeError("Network unknown")

        saved_state_dict   = torch.load(weights[i], map_location=device)
        model.load_state_dict(saved_state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)

    if _config["save_name"] is not None or _config['save_log']:
        if os.path.isdir('./results_for_paper'):
            shutil.rmtree("./results_for_paper")
        os.mkdir("./results_for_paper")

    if _config['save_log']:
        log_file = f'./results_for_paper/log_seq{_config["test_sequence"]}.csv'
        log_file = open(log_file, 'w')
        log_file = csv.writer(log_file)
        header = ['frame']
        for i in range(len(weights) + 1):
            header += [f'iter{i}_error_t', f'iter{i}_error_r', f'iter{i}_error_x', f'iter{i}_error_y',
                       f'iter{i}_error_z', f'iter{i}_error_r', f'iter{i}_error_p', f'iter{i}_error_y']
        log_file.writerow(header)

    show = _config['show']
    # save image to the output path
    output_dir = os.path.join(_config['output'], _config['iterative_method'])
    rgb_path = os.path.join(output_dir, 'rgb')
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    depth_path = os.path.join(output_dir, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    input_path = os.path.join(output_dir, 'input')
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    gt_path = os.path.join(output_dir, 'gt')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if _config['out_fig_lg'] == 'EN':
        results_path = os.path.join(output_dir, 'results_en')
    else:
        results_path = os.path.join(output_dir, 'results_cn')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    pred_path = os.path.join(output_dir, 'pred')
    for it in range(len(weights)):
        if not os.path.exists(os.path.join(pred_path, 'iteration_'+str(it+1))):
            os.makedirs(os.path.join(pred_path, 'iteration_'+str(it+1)))

    # save pointcloud to the output path
    pc_lidar_path = os.path.join(_config['output'], 'pointcloud', 'lidar')
    if not os.path.exists(pc_lidar_path):
        os.makedirs(pc_lidar_path)
    pc_input_path = os.path.join(_config['output'], 'pointcloud', 'input')
    if not os.path.exists(pc_input_path):
        os.makedirs(pc_input_path)
    pc_pred_path = os.path.join(_config['output'], 'pointcloud', 'pred')
    if not os.path.exists(pc_pred_path):
        os.makedirs(pc_pred_path)


    errors_r = []
    errors_t = []
    errors_t2 = []
    errors_xyz = []
    errors_rpy = []
    all_RTs = []
    mis_calib_list = []
    total_time = 0

    prev_tr_error = None
    prev_rot_error = None

    for i in range(len(weights) + 1):
        errors_r.append([])
        errors_t.append([])
        errors_t2.append([])
        errors_rpy.append([])

    mean_rpe_list = []
    # Loop over each frame in the selected test sequence
    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        print("\n")
        N = 100 # 500
        # if batch_idx > 200:
        #    break

        log_string = [str(batch_idx)]

        lidar_input = []
        rgb_input = []
        lidar_gt = []
        shape_pad_input = []
        real_shape_input = []
        pc_rotated_input = []
        RTs = []
        shape_pad = [0, 0, 0, 0]
        outlier_filter = False

        if batch_idx == 0 or not _config['use_prev_output']:
            # Qui dare posizione di input del frame corrente rispetto alla GT
            sample['tr_error'] = sample['tr_error'].to(device)
            sample['rot_error'] = sample['rot_error'].to(device)
        else:
            sample['tr_error'] = prev_tr_error
            sample['rot_error'] = prev_rot_error

        # For each RGB‐LiDAR pair in the sample: 
        # project the raw and perturbed point clouds into image space to produce normalized depth maps, 
        # apply optional outlier filtering, and (if enabled) save both the raw and input point clouds as PCD files.
        for idx in range(len(sample['rgb'])):
            # ProjectPointCloud in RT-pose
            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].to(device)
            pc_lidar = sample['point_cloud'][idx].clone()

            if _config['max_depth'] < 80.:
                pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

            depth_gt, uv_gt, pc_gt_valid = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape)  # image_shape
            depth_gt /= _config['max_depth']

            if _config['save_image']:
                # save the Lidar pointcloud
                pcl_lidar = o3.geometry.PointCloud()
                pc_lidar = pc_lidar.detach().cpu().numpy()
                pcl_lidar.points = o3.utility.Vector3dVector(pc_lidar.T[:, :3])

                o3.io.write_point_cloud(pc_lidar_path + '/{}.pcd'.format(batch_idx), pcl_lidar)

            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT_inv = torch.mm(T, R)
            RT = RT_inv.clone().inverse()

            pc_rotated = rotate_back(sample['point_cloud'][idx], RT_inv)  # Pc` = RT * Pc

            if _config['max_depth'] < 80.:
                pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

            depth_img, uv_input, pc_input_valid = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape)  # image_shape
            depth_img /= _config['max_depth']

            if _config['outlier_filter'] and uv_input.shape[0] <= _config['outlier_filter_th']:
                outlier_filter = True
            else:
                outlier_filter = False

            if _config['save_image']:
                # save the RGB input pointcloud
                img = cv2.imread(sample['img_path'][0])
                R = img[uv_input[:, 1], uv_input[:, 0], 0] / 255
                G = img[uv_input[:, 1], uv_input[:, 0], 1] / 255
                B = img[uv_input[:, 1], uv_input[:, 0], 2] / 255
                pcl_input = o3.geometry.PointCloud()
                pcl_input.points = o3.utility.Vector3dVector(pc_input_valid[:, :3])
                pcl_input.colors = o3.utility.Vector3dVector(np.vstack((R, G, B)).T)

                o3.io.write_point_cloud(pc_input_path + '/{}.pcd'.format(batch_idx), pcl_input)

            # PAD ONLY ON RIGHT AND BOTTOM SIDE
            rgb = sample['rgb'][idx].to(device)
            shape_pad = [0, 0, 0, 0]

            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            rgb = F.pad(rgb, shape_pad)
            depth_img = F.pad(depth_img, shape_pad)
            depth_gt = F.pad(depth_gt, shape_pad)

            rgb_input.append(rgb)
            lidar_input.append(depth_img)
            lidar_gt.append(depth_gt)
            real_shape_input.append(real_shape)
            shape_pad_input.append(shape_pad)
            pc_rotated_input.append(pc_rotated)
            RTs.append(RT)

        if outlier_filter:
            continue

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)
        rgb_resize = F.interpolate(rgb_input, size=input_size, mode="bilinear")
        lidar_resize = F.interpolate(lidar_input, size=input_size, mode="bilinear")

        if _config['save_image']:
            out0 = overlay_imgs(rgb_input[0], lidar_input)
            out0 = out0[:376, :1241, :]
            cv2.imwrite(os.path.join(input_path, sample['rgb_name'][0]), out0[:, :, [2, 1, 0]]*255)
            out1 = overlay_imgs(rgb_input[0], lidar_gt[0].unsqueeze(0))
            out1 = out1[:376, :1241, :]
            cv2.imwrite(os.path.join(gt_path, sample['rgb_name'][0]), out1[:, :, [2, 1, 0]]*255)

            depth_img = depth_img.detach().cpu().numpy()
            depth_img = (depth_img / np.max(depth_img)) * 255
            orig_h, orig_w = sample['rgb'][0].shape[1], sample['rgb'][0].shape[2]
            cv2.imwrite(os.path.join(depth_path, sample['rgb_name'][0]), depth_img[0, :orig_h, :orig_w])

        if show:
            out0 = overlay_imgs(rgb_input[0], lidar_input)
            out1 = overlay_imgs(rgb_input[0], lidar_gt[0].unsqueeze(0))
            cv2.imshow("INPUT", out0[:, :, [2, 1, 0]])
            cv2.imshow("GT", out1[:, :, [2, 1, 0]])
            cv2.waitKey(1)

        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)
        rgb_resize = rgb_resize.to(device)
        lidar_resize = lidar_resize.to(device)

        target_transl = sample['tr_error'].to(device)
        target_rot = sample['rot_error'].to(device)

        # the initial calibration errors before sensor calibration
        RT1 = RTs[0]
        mis_calib = torch.stack(sample['initial_RT'])[1:]
        mis_calib_list.append(mis_calib)

        T_composed = RT1[:3, 3]
        R_composed = quaternion_from_matrix(RT1)
        errors_t[0].append(T_composed.norm().item())
        errors_t2[0].append(T_composed)
        errors_r[0].append(quaternion_distance(R_composed.unsqueeze(0),
                                               torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                               R_composed.device))
        # rpy_error = quaternion_to_tait_bryan(R_composed)
        rpy_error = mat2xyzrpy(RT1)[3:]

        rpy_error *= (180.0 / 3.141592)
        errors_rpy[0].append(rpy_error)
        log_string += [str(errors_t[0][-1]), str(errors_r[0][-1]), str(errors_t2[0][-1][0].item()),
                       str(errors_t2[0][-1][1].item()), str(errors_t2[0][-1][2].item()),
                       str(errors_rpy[0][-1][0].item()), str(errors_rpy[0][-1][1].item()),
                       str(errors_rpy[0][-1][2].item())]

        print(f"Frame {batch_idx:04d} Calibration Errors per Iteration:")
        for it in range(1, len(weights) + 1):
            t_raw = errors_t[it][-1] if errors_t[it] else float('nan')
            r_raw = errors_r[it][-1].item() if errors_r[it] else float('nan')
            # Convert to user-friendly units
            t_err_cm  = t_raw * 100.0            # m --> cm
            r_err_deg = r_raw * 180.0 / np.pi    # rad --> °
            print(f"  Iteration {it}:  Translation Error: {t_err_cm:.4f} cm    "
                  f"Rotation Error: {r_err_deg:.4f} °"
            )

        # if batch_idx == 0.:
        #     print(f'Initial T_erorr: {errors_t[0]}')
        #     print(f'Initial R_erorr: {errors_r[0]}')
        start = 0
        # t1 = time.time()

        init_extr = sample['extrin'][0]
        if isinstance(init_extr, np.ndarray):
            H_init = torch.from_numpy(init_extr).float().to(device)
        else:
            H_init = init_extr.to(device)

        '''
        # Debug: If you apply translation changes to your pointcloud, you can see in which dimension they are with which sign in the translation vector:
        # 1. Extract the 3x3 rotation R_LC
        R_LC = H_init[:3, :3]  # shape (3,3)

        changes_xyz = [0.75, 1.50, 0.00]

        # 2. Build the LiDAR-frame shift vector
        d_L = torch.tensor(changes_xyz, device=R_LC.device)  # shape (3,)

        # 3. Compute Δt = -R_LC @ d_L
        delta_t = - (R_LC @ d_L)  # shape (3,)

        # 4. (optional) move back to CPU + NumPy for printing or further use
        delta_t_np = delta_t.detach().cpu().numpy()
        print("Expected Δt (camera frame):", delta_t_np)
        '''

        # Run model
        with torch.no_grad():
            # Make the predictions with each model and refine the results
            for iteration in range(start, len(weights)):
                # Run the i-th network
                t1 = time.time()
                if _config['iterative_method'] == 'single_range' or _config['iterative_method'] == 'single':
                    T_predicted, R_predicted = models[0](rgb_resize, lidar_resize)
                elif _config['iterative_method'] == 'multi_range':
                    T_predicted, R_predicted = models[iteration](rgb_resize, lidar_resize)
                run_time = time.time() - t1

                if _config['rot_transl_separated'] and iteration == 0:
                    T_predicted = torch.tensor([[0., 0., 0.]], device=device)
                if _config['rot_transl_separated'] and iteration == 1:
                    R_predicted = torch.tensor([[1., 0., 0., 0.]], device=device)

                # Convert quaternion and translation vector into 4×4 matrices
                R_mat = quat2mat(R_predicted[0])      # 4×4 rotation
                T_mat = tvector2mat(T_predicted[0])    # 4×4 translation

                # Compose with previous pose (cumulative transform)
                RT_predicted = torch.mm(T_mat, R_mat)    # ΔH (network residual)
                RTs.append(torch.mm(RTs[iteration], RT_predicted))
                # Compute absolute extrinsic using cumulative transform per Formula (10)
                H_abs = torch.inverse(RTs[iteration + 1]) @ H_init

                # Output of predicted extrinsic change and absolute extrinsic
                print(f"Frame {batch_idx:04d}, Iter {iteration+1} Predicted RT:")
                print(RTs[iteration+1].cpu().numpy())

                print(f"Frame {batch_idx}, Iter {iteration+1}, Absolute Extrinsic:")
                print(H_abs.cpu().numpy())

                # Extract rotation matrix and translation vector from H_abs
                H_abs_np = H_abs.cpu().numpy()
                R_abs = H_abs_np[:3, :3]
                tvec_abs = H_abs_np[:3, 3].reshape(3, 1)

                # Convert rotation matrix to Rodrigues rotation vector
                rvec_abs, _ = cv2.Rodrigues(R_abs)

                # Print tvec and rvec
                print(f"Frame {batch_idx:04d}, Iter {iteration+1} Absolute Translation vector (tvec):")
                print(tvec_abs)
                print(f"Frame {batch_idx:04d}, Iter {iteration+1} Absolute Rotation vector (rvec):")
                print(rvec_abs)

                # Start reprojection error computation after last iteration
                if iteration == len(weights) - 1:

                    pc_lidar_np = pc_lidar.T[:, :3]  # (N, 3)
                    cam_intrinsic = sample['calib'][0].cpu().numpy()
                    real_shape = real_shape_input[0]

                    N_pts = pc_lidar_np.shape[0]

                    # project ground-truth
                    H_gt_np = sample['extrin'][0]
                    if isinstance(H_gt_np, torch.Tensor):
                        H_gt_np = H_gt_np.cpu().numpy()
                    pcl_gt_proj, pcl_gt_z = project_pointcloud(pc_lidar_np, H_gt_np, cam_intrinsic)

                    padded_H, padded_W = img_shape
                    mask_gt = (
                        (pcl_gt_proj[:,0] > 0)  & (pcl_gt_proj[:,0] < padded_W) &
                        (pcl_gt_proj[:,1] > 0)  & (pcl_gt_proj[:,1] < padded_H) &
                        (pcl_gt_z   > 0)
                    )

                    # project estimated
                    H_abs_np = H_abs.cpu().numpy()
                    pcl_est_proj, pcl_est_z = project_pointcloud(pc_lidar_np, H_abs_np, cam_intrinsic)
                    mask_est = (
                        (pcl_est_proj[:,0] > 0)  & (pcl_est_proj[:,0] < padded_W) &
                        (pcl_est_proj[:,1] > 0)  & (pcl_est_proj[:,1] < padded_H) &
                        (pcl_est_z   > 0)
                    )

                    # joint mask
                    mask_joint = mask_gt & mask_est

                    # print debug info
                    img_file = os.path.basename(sample['img_path'][0])
                    print(f"[{img_file}] Frame {batch_idx:04d} RPE debug — "
                        f"total_pts={N_pts}, GT_in={mask_gt.sum()}, EST_in={mask_est.sum()}, joint={mask_joint.sum()}")

                    # ranges
                    if mask_gt.sum() > 0:
                        u_gt, v_gt = pcl_gt_proj[mask_gt, 0], pcl_gt_proj[mask_gt, 1]
                        print(f"  GT u: {u_gt.min():.1f}-{u_gt.max():.1f}, v: {v_gt.min():.1f}-{v_gt.max():.1f}")
                    if mask_est.sum() > 0:
                        u_e, v_e = pcl_est_proj[mask_est, 0], pcl_est_proj[mask_est, 1]
                        print(f"  EST u: {u_e.min():.1f}-{u_e.max():.1f}, v: {v_e.min():.1f}-{v_e.max():.1f}")

                    pcl_gt_proj_valid = pcl_gt_proj[mask_joint]
                    pcl_est_proj_valid = pcl_est_proj[mask_joint]

                    if pcl_gt_proj_valid.shape[0] == 0:
                        print(f"Frame {batch_idx:04d}: No valid points for reprojection error.")
                        mean_rpe_list.append(-1)
                    else:
                        reprojection_errors = np.linalg.norm(pcl_gt_proj_valid - pcl_est_proj_valid, axis=1)
                        rpe_mean = np.mean(reprojection_errors)
                        mean_rpe_list.append(rpe_mean)
                        print(f"Frame {batch_idx:04d}: Mean Reprojection Error = {rpe_mean:.4f} pixels")

                # Project the points in the new pose predicted by the i-th network
                R_predicted = quat2mat(R_predicted[0])
                T_predicted = tvector2mat(T_predicted[0])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                #RTs.append(torch.mm(RTs[iteration], RT_predicted)) # inv(H_gt)*H_pred_1*H_pred_2*.....H_pred_n  # Already done above!
                if iteration == 0:
                    rotated_point_cloud = pc_rotated_input[0]
                else:
                    rotated_point_cloud = rotated_point_cloud

                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted) # H_pred*X_init

                depth_img_pred, uv_pred, pc_pred_valid = lidar_project_depth(rotated_point_cloud, sample['calib'][0], real_shape_input[0]) # image_shape
                depth_img_pred /= _config['max_depth']
                depth_pred = F.pad(depth_img_pred, shape_pad_input[0])
                lidar = depth_pred.unsqueeze(0)
                lidar_resize = F.interpolate(lidar, size=input_size, mode="bilinear")

                if iteration == len(weights)-1 and _config['save_image']:
                    # save the RGB pointcloud
                    img = cv2.imread(sample['img_path'][0])
                    R = img[uv_pred[:, 1], uv_pred[:, 0], 0] / 255
                    G = img[uv_pred[:, 1], uv_pred[:, 0], 1] / 255
                    B = img[uv_pred[:, 1], uv_pred[:, 0], 2] / 255
                    pcl_pred = o3.geometry.PointCloud()
                    pcl_pred.points = o3.utility.Vector3dVector(pc_pred_valid[:, :3])
                    pcl_pred.colors = o3.utility.Vector3dVector(np.vstack((R, G, B)).T)

                    o3.io.write_point_cloud(pc_pred_path + '/{}.pcd'.format(batch_idx), pcl_pred)

                if _config['save_image']:
                    out2 = overlay_imgs(rgb_input[0], lidar)
                    out2 = out2[:376, :1241, :]
                    cv2.imwrite(os.path.join(os.path.join(pred_path, 'iteration_'+str(iteration+1)),
                                             sample['rgb_name'][0]), out2[:, :, [2, 1, 0]]*255)
                if show:
                    out2 = overlay_imgs(rgb_input[0], lidar)
                    cv2.imshow(f'Pred_Iter_{iteration}', out2[:, :, [2, 1, 0]])
                    cv2.waitKey(1)

                # inv(H_init)*H_pred
                T_composed = RTs[iteration + 1][:3, 3]
                R_composed = quaternion_from_matrix(RTs[iteration + 1])
                errors_t[iteration + 1].append(T_composed.norm().item())
                errors_t2[iteration + 1].append(T_composed)
                errors_r[iteration + 1].append(quaternion_distance(R_composed.unsqueeze(0),
                                                                   torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                                                   R_composed.device))

                # rpy_error = quaternion_to_tait_bryan(R_composed)
                rpy_error = mat2xyzrpy(RTs[iteration + 1])[3:]
                rpy_error *= (180.0 / 3.141592)
                errors_rpy[iteration + 1].append(rpy_error)
                log_string += [str(errors_t[iteration + 1][-1]), str(errors_r[iteration + 1][-1]),
                               str(errors_t2[iteration + 1][-1][0].item()), str(errors_t2[iteration + 1][-1][1].item()),
                               str(errors_t2[iteration + 1][-1][2].item()), str(errors_rpy[iteration + 1][-1][0].item()),
                               str(errors_rpy[iteration + 1][-1][1].item()), str(errors_rpy[iteration + 1][-1][2].item())]

        # run_time = time.time() - t1
        total_time += run_time

        # final calibration error
        all_RTs.append(RTs[-1])
        prev_RT = RTs[-1].inverse()
        prev_tr_error = prev_RT[:3, 3].unsqueeze(0)
        prev_rot_error = quaternion_from_matrix(prev_RT).unsqueeze(0)

        if _config['save_log']:
            log_file.writerow(log_string)

    # Yaw（偏航）：欧拉角向量的y轴
    # Pitch（俯仰）：欧拉角向量的x轴
    # Roll（翻滚）： 欧拉角向量的z轴
    # mis_calib_input[transl_x, transl_y, transl_z, rotx, roty, rotz] Nx6
    if not mis_calib_list:
        raise RuntimeError("mis_calib_list is empty. "
                           "Either disable outlier_filter or verify that your dataset produces valid "
                           "projected point clouds (uv_input)."
    )
    mis_calib_input = torch.stack(mis_calib_list)[:, :, 0]

    if _config['save_log']:
        log_file.close()

    # write out all per-frame means and their average
    mean_file = os.path.join(_config['output'], "mean_projection_errors.txt")
    with open(mean_file, "w", encoding="utf-8") as f:
        for val in mean_rpe_list:
            if val != -1:
                f.write(f"{val:.4f}\n")
            else:
                f.write("No valid points for reprojection error.\n")
        temp = [v for v in mean_rpe_list if v != -1]
        avg_rpe = float(np.mean(temp))
        f.write(f"\nMPE average of {len(temp)} pairs: {avg_rpe:.4f}\n")

    print("\nIterative refinement: ")
    for i in range(len(weights) + 1):
        errors_r[i] = torch.tensor(errors_r[i]).abs() * (180.0 / 3.141592)
        errors_t[i] = torch.tensor(errors_t[i]).abs() * 100

        for k in range(len(errors_rpy[i])):
            # errors_rpy[i][k] = torch.tensor(errors_rpy[i][k])
            # errors_t2[i][k] = torch.tensor(errors_t2[i][k]) * 100
            errors_rpy[i][k] = errors_rpy[i][k].clone().detach().abs()
            errors_t2[i][k] = errors_t2[i][k].clone().detach().abs() * 100

        header = "Baseline (initial calibration - no refinement applied)" if i == 0 else f"Iteration {i}"
        print(f"{header}:")
        print("Translation Error && Rotation Error:")
        print(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
              f"     Mean Rotation Error: {errors_r[i].mean():.4f} °")
        print(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
              f"     Median Rotation Error: {errors_r[i].median():.4f} °")
        print(f"Iteration {i}: \tStd. Translation Error: {errors_t[i].std():.4f} cm "
              f"     Std. Rotation Error: {errors_r[i].std():.4f} °\n")

        # translation xyz
        print("Translation Error XYZ:")
        print(f"Iteration {i}: \tMean Translation X Error: {errors_t2[i][0].mean():.4f} cm "
              f"     Median Translation X Error: {errors_t2[i][0].median():.4f} cm "
              f"     Std. Translation X Error: {errors_t2[i][0].std():.4f} cm ")
        print(f"Iteration {i}: \tMean Translation Y Error: {errors_t2[i][1].mean():.4f} cm "
              f"     Median Translation Y Error: {errors_t2[i][1].median():.4f} cm "
              f"     Std. Translation Y Error: {errors_t2[i][1].std():.4f} cm ")
        print(f"Iteration {i}: \tMean Translation Z Error: {errors_t2[i][2].mean():.4f} cm "
              f"     Median Translation Z Error: {errors_t2[i][2].median():.4f} cm "
              f"     Std. Translation Z Error: {errors_t2[i][2].std():.4f} cm \n")

        # rotation rpy
        print("Rotation Error RPY:")
        print(f"Iteration {i}: \tMean Rotation Roll Error: {errors_rpy[i][0].mean(): .4f} °"
              f"     Median Rotation Roll Error: {errors_rpy[i][0].median():.4f} °"
              f"     Std. Rotation Roll Error: {errors_rpy[i][0].std():.4f} °")
        print(f"Iteration {i}: \tMean Rotation Pitch Error: {errors_rpy[i][1].mean(): .4f} °"
              f"     Median Rotation Pitch Error: {errors_rpy[i][1].median():.4f} °"
              f"     Std. Rotation Pitch Error: {errors_rpy[i][1].std():.4f} °")
        print(f"Iteration {i}: \tMean Rotation Yaw Error: {errors_rpy[i][2].mean(): .4f} °"
              f"     Median Rotation Yaw Error: {errors_rpy[i][2].median():.4f} °"
              f"     Std. Rotation Yaw Error: {errors_rpy[i][2].std():.4f} °\n")


        with open(os.path.join(_config['output'], 'results.txt'),
                  'a', encoding='utf-8') as f:
            header = "Baseline (initial calibration – no refinement applied)" if i == 0 else f"Iteration {i}"
            f.write(f"{header}:\n")
            f.write("Translation Error && Rotation Error:\n")
            f.write(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
                    f"     Mean Rotation Error: {errors_r[i].mean():.4f} °\n")
            f.write(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
                    f"     Median Rotation Error: {errors_r[i].median():.4f} °\n")
            f.write(f"Iteration {i}: \tStd. Translation Error: {errors_t[i].std():.4f} cm "
                    f"     Std. Rotation Error: {errors_r[i].std():.4f} °\n\n")

            # translation xyz
            f.write("Translation Error XYZ:\n")
            f.write(f"Iteration {i}: \tMean Translation X Error: {errors_t2[i][0].mean():.4f} cm "
                    f"     Median Translation X Error: {errors_t2[i][0].median():.4f} cm "
                    f"     Std. Translation X Error: {errors_t2[i][0].std():.4f} cm \n")
            f.write(f"Iteration {i}: \tMean Translation Y Error: {errors_t2[i][1].mean():.4f} cm "
                    f"     Median Translation Y Error: {errors_t2[i][1].median():.4f} cm "
                    f"     Std. Translation Y Error: {errors_t2[i][1].std():.4f} cm \n")
            f.write(f"Iteration {i}: \tMean Translation Z Error: {errors_t2[i][2].mean():.4f} cm "
                    f"     Median Translation Z Error: {errors_t2[i][2].median():.4f} cm "
                    f"     Std. Translation Z Error: {errors_t2[i][2].std():.4f} cm \n\n")

            # rotation rpy
            f.write("Rotation Error RPY:\n")
            f.write(f"Iteration {i}: \tMean Rotation Roll Error: {errors_rpy[i][0].mean(): .4f} °"
                    f"     Median Rotation Roll Error: {errors_rpy[i][0].median():.4f} °"
                    f"     Std. Rotation Roll Error: {errors_rpy[i][0].std():.4f} °\n")
            f.write(f"Iteration {i}: \tMean Rotation Pitch Error: {errors_rpy[i][1].mean(): .4f} °"
                    f"     Median Rotation Pitch Error: {errors_rpy[i][1].median():.4f} °"
                    f"     Std. Rotation Pitch Error: {errors_rpy[i][1].std():.4f} °\n")
            f.write(f"Iteration {i}: \tMean Rotation Yaw Error: {errors_rpy[i][2].mean(): .4f} °"
                    f"     Median Rotation Yaw Error: {errors_rpy[i][2].median():.4f} °"
                    f"     Std. Rotation Yaw Error: {errors_rpy[i][2].std():.4f} °\n\n\n")

    for i in range(len(errors_t2)):
        errors_t2[i] = torch.stack(errors_t2[i]).abs() / 100
        errors_rpy[i] = torch.stack(errors_rpy[i]).abs()

    # mis_calib_input
    # t_x = mis_calib_input[:, 0]
    # t_y = mis_calib_input[:, 1]
    # t_z = mis_calib_input[:, 2]
    # r_roll = mis_calib_input[:, 5]
    # r_pitch = mis_calib_input[:, 3]
    # r_yaw = mis_calib_input[:, 4]

    # plot_error
    # plot_x = errors_t2[:, 0]
    # plot_y = errors_t2[:, 1]
    # plot_z = errors_t2[:, 2]
    # plot_roll = errors_rpy[:, 0]
    # plot_pitch = errors_rpy[:, 1]
    # plot_yaw = errors_rpy[:, 2]

    # translation error
    # fig = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
    # plt.title('Calibration Translation Error')
    plot_x = np.zeros((mis_calib_input.shape[0], 2))
    plot_x[:, 0] = mis_calib_input[:, 0].cpu().numpy()
    plot_x[:, 1] = errors_t2[-1][:, 0].cpu().numpy()
    plot_x = plot_x[np.lexsort(plot_x[:, ::-1].T)]

    plot_y = np.zeros((mis_calib_input.shape[0], 2))
    plot_y[:, 0] = mis_calib_input[:, 1].cpu().numpy()
    plot_y[:, 1] = errors_t2[-1][:, 1].cpu().numpy()
    plot_y = plot_y[np.lexsort(plot_y[:, ::-1].T)]

    plot_z = np.zeros((mis_calib_input.shape[0], 2))
    plot_z[:, 0] = mis_calib_input[:, 2].cpu().numpy()
    plot_z[:, 1] = errors_t2[-1][:, 2].cpu().numpy()
    plot_z = plot_z[np.lexsort(plot_z[:, ::-1].T)]

    if not plot_x.shape[0] // N:
        N = plot_x.shape[0] // 5

    N_interval = plot_x.shape[0] // N
    plot_x = plot_x[::N_interval]
    plot_y = plot_y[::N_interval]
    plot_z = plot_z[::N_interval]

    plt.plot(plot_x[:, 0], plot_x[:, 1], c='red', label='X')
    plt.plot(plot_y[:, 0], plot_y[:, 1], c='blue', label='Y')
    plt.plot(plot_z[:, 0], plot_z[:, 1], c='green', label='Z')
    # plt.legend(loc='best')

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Miscalibration (m)', font_EN)
        plt.ylabel('Absolute Error (m)', font_EN)
        plt.legend(loc='best', prop=font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('初始标定外参偏差/米', font_CN)
        plt.ylabel('绝对误差/米', font_CN)
        plt.legend(loc='best', prop=font_CN)

    plt.xticks(size=plt_size)
    plt.yticks(size=plt_size)

    plt.savefig(os.path.join(results_path, 'xyz_plot.png'))
    plt.close('all')

    errors_t_np = errors_t[-1].cpu().numpy()
    errors_t_np = np.sort(errors_t_np, axis=0)[:-10]
    errors_t = errors_t[-1].numpy()
    errors_t = np.sort(errors_t, axis=0)[:-10] # 去掉一些异常值
    # plt.title('Calibration Translation Error Distribution')
    plt.hist(errors_t / 100, bins=50)
    # ax = plt.gca()
    # ax.set_xlabel('Absolute Translation Error (m)')
    # ax.set_ylabel('Number of instances')
    # ax.set_xticks([0.00, 0.25, 0.00, 0.25, 0.50])

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Absolute Translation Error (m)', font_EN)
        plt.ylabel('Number of instances', font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('绝对平移误差/米', font_CN)
        plt.ylabel('实验序列数目/个', font_CN)
    plt.xticks(size=plt_size)
    plt.yticks(size=plt_size)

    plt.savefig(os.path.join(results_path, 'translation_error_distribution.png'))
    plt.close('all')

    # rotation error
    # fig = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
    # plt.title('Calibration Rotation Error')
    plot_pitch = np.zeros((mis_calib_input.shape[0], 2))
    plot_pitch[:, 0] = mis_calib_input[:, 3].cpu().numpy() * (180.0 / 3.141592)
    plot_pitch[:, 1] = errors_rpy[-1][:, 1].cpu().numpy()
    plot_pitch = plot_pitch[np.lexsort(plot_pitch[:, ::-1].T)]

    plot_yaw = np.zeros((mis_calib_input.shape[0], 2))
    plot_yaw[:, 0] = mis_calib_input[:, 4].cpu().numpy() * (180.0 / 3.141592)
    plot_yaw[:, 1] = errors_rpy[-1][:, 2].cpu().numpy()
    plot_yaw = plot_yaw[np.lexsort(plot_yaw[:, ::-1].T)]

    plot_roll = np.zeros((mis_calib_input.shape[0], 2))
    plot_roll[:, 0] = mis_calib_input[:, 5].cpu().numpy() * (180.0 / 3.141592)
    plot_roll[:, 1] = errors_rpy[-1][:, 0].cpu().numpy()
    plot_roll = plot_roll[np.lexsort(plot_roll[:, ::-1].T)]

    N_interval = plot_roll.shape[0] // N
    plot_pitch = plot_pitch[::N_interval]
    plot_yaw = plot_yaw[::N_interval]
    plot_roll = plot_roll[::N_interval]

    # Yaw（偏航）：欧拉角向量的y轴
    # Pitch（俯仰）：欧拉角向量的x轴
    # Roll（翻滚）： 欧拉角向量的z轴

    if _config['out_fig_lg'] == 'EN':
        plt.plot(plot_yaw[:, 0], plot_yaw[:, 1], c='red', label='Yaw(Y)')
        plt.plot(plot_pitch[:, 0], plot_pitch[:, 1], c='blue', label='Pitch(X)')
        plt.plot(plot_roll[:, 0], plot_roll[:, 1], c='green', label='Roll(Z)')
        plt.xlabel('Miscalibration (°)', font_EN)
        plt.ylabel('Absolute Error (°)', font_EN)
        plt.legend(loc='best', prop=font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.plot(plot_yaw[:, 0], plot_yaw[:, 1], c='red', label='偏航角')
        plt.plot(plot_pitch[:, 0], plot_pitch[:, 1], c='blue', label='俯仰角')
        plt.plot(plot_roll[:, 0], plot_roll[:, 1], c='green', label='翻滚角')
        plt.xlabel('初始标定外参偏差/度', font_CN)
        plt.ylabel('绝对误差/度', font_CN)
        plt.legend(loc='best', prop=font_CN)

    plt.xticks(size=plt_size)
    plt.yticks(size=plt_size)
    plt.savefig(os.path.join(results_path, 'rpy_plot.png'))
    plt.close('all')

    errors_r_np = errors_r[-1].cpu().numpy()
    errors_r_np = np.sort(errors_r_np, axis=0)[:-10]
    errors_r = errors_r[-1].numpy()
    errors_r = np.sort(errors_r, axis=0)[:-10] # 去掉一些异常值
    # np.savetxt('rot_error.txt', arr_, fmt='%0.8f')
    # print('max rotation_error: {}'.format(max(errors_r)))
    # plt.title('Calibration Rotation Error Distribution')
    plt.hist(errors_r, bins=50)
    #plt.xlim([0, 1.5])  # x轴边界
    #plt.xticks([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])  # 设置x刻度
    # ax = plt.gca()

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Absolute Rotation Error (°)', font_EN)
        plt.ylabel('Number of instances', font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('绝对旋转误差/度', font_CN)
        plt.ylabel('实验序列数目/个', font_CN)
    plt.xticks(size=plt_size)
    plt.yticks(size=plt_size)
    plt.savefig(os.path.join(results_path, 'rotation_error_distribution.png'))
    plt.close('all')


    if _config["save_name"] is not None:
        np.save(f"./results_for_paper/{_config['save_name']}_errors_t_np.npy", errors_t_np)
        np.save(f"./results_for_paper/{_config['save_name']}_errors_r_np.npy", errors_r_np)
        torch.save(torch.stack(errors_t2).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_t2.pt')
        torch.save(torch.stack(errors_rpy).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_rpy.pt')

    avg_time = total_time / len(TestImgLoader)
    print("Average running time on {} iteration: {} s".format(len(weights), avg_time))
    print("End!")
