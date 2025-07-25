# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/main_visibility_CALIB.py

import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetLidarCamera import DatasetLidarCameraKittiOdometry
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
from models.LCCNet import LCCNet

from quaternion_distances import quaternion_distance

from tensorboardX import SummaryWriter
from utils import (merge_inputs, overlay_imgs, quat2mat,
                   rotate_back, rotate_forward,
                   tvector2mat)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

ex = Experiment("LCCNet", save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = './checkpoints/'
    dataset = 'kitti/odom' # 'kitti/raw'
    data_folder = './datasets/own_data2_mixed_scenarios'  # './datasets/odometry_color_short/'
    #data_folder = './datasets/odometry_color_short/'
    img_shape  = (2128, 2600)  # padded image resolution (H, W)  # KITTI: (384, 1280)  # Own: (2128, 2600)
    #img_shape  = (384, 1280)
    input_size  = (256, 512)  # network input resolution (H, W)  # KITTI: (256, 512)  # Own: (256, 512)
    use_reflectance = False
    val_sequence = 0
    epochs = 50  # 120 for the first model (iter1), every other only 50, since we can use the previous iteration model as a pretrained model for the next one
    BASE_LEARNING_RATE = 3e-4  # 1e-4
    loss = 'combined'
    max_t = 1.5/2.0  # iter1, iter2, 3, 4, 5: 1.5, 1.0, 0.5, 0.2, 0.1        # 0.75, 0.5
    max_r = 20.0/2.0  # iter1, iter2, 3, 4, 5: 20.0, 10.0, 5.0, 2.0, 1.0      # 10.0, 5.0
    batch_size = 32
    num_worker = 6
    network = 'Res_f1'
    optimizer = 'adam'
    resume = False
    weights = 'checkpoints/kitti/odom/val_seq_00/models/checkpoint_r10.00_t0.75_e280_0.177.pth'  # './pretrained/kitti_iter5.tar'  # set a weights file to use a pretrained one or None for a start from scratch
    rescale_rot = 1.0
    rescale_transl = 2.0
    precision = "O0"
    norm = 'bn'
    dropout = 0.0
    max_depth = 50.0
    weight_point_cloud = 0.5
    log_frequency = 10
    print_frequency = 50
    starting_epoch = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

max_depth_behaviour_switch = 50.1


EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    # print(f"Init worker {worker_id} with seed {seed}")
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
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.to(device)
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv


# CCN training
@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.train()

    optimizer.zero_grad()

    # Run model
    transl_err, rot_err = model(rgb_img, refl_img)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    losses['total_loss'].backward()
    optimizer.step()

    # print("rgb_img, refl_img:", rgb_img, refl_img)
    # print("non-zero values of rgb_img:", torch.count_nonzero(rgb_img))
    # print("non-zero values of refl_img:", torch.count_nonzero(refl_img))

    # print("target_transl, target_rot:", target_transl, target_rot)
    # print("transl_err, rot_err", transl_err, rot_err)
    # print("loss_fn, point_clouds:", loss_fn, point_clouds)

    return losses, rot_err, transl_err


# CNN test
@ex.capture
def val(model, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.eval()

    # Run model
    with torch.no_grad():
        transl_err, rot_err = model(rgb_img, refl_img)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    # if loss != 'points_distance':
    #     total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    # else:
    #     total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_transl_error = torch.tensor(0.0, device=target_transl.device)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_transl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    # # output image: The overlay image of the input rgb image and the projected lidar pointcloud depth image
    # cam_intrinsic = camera_model[0]
    # rotated_point_cloud =
    # R_predicted = quat2mat(R_predicted[0])
    # T_predicted = tvector2mat(T_predicted[0])
    # RT_predicted = torch.mm(T_predicted, R_predicted)
    # rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

    return losses, total_transl_error.item(), total_rot_error.sum().item(), rot_err, transl_err


@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))

    if _config['val_sequence'] is None:
        raise TypeError('val_sequences cannot be None')
    else:
        if isinstance(_config['val_sequence'], int):
            val_sequence = f"{_config['val_sequence']:02d}"
        else:
            val_sequence = _config['val_sequence']
        print("Val Sequence: ", val_sequence)
        dataset_class = DatasetLidarCameraKittiOdometry

    img_shape   = tuple(_config['img_shape'])
    input_size  = tuple(_config['input_size'])

    checkpoint_root = os.path.join(_config["checkpoints"], _config['dataset'])

    dataset_train = dataset_class(
        _config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
        split='train', use_reflectance=_config['use_reflectance'],
        val_sequence=val_sequence,
        device=device
    )
    dataset_val = dataset_class(
       _config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
        split='val', use_reflectance=_config['use_reflectance'],
        val_sequence=val_sequence,
        device=device
    )

    model_savepath = os.path.join(checkpoint_root, f'val_seq_{val_sequence}', 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    log_savepath = os.path.join(checkpoint_root, f'val_seq_{val_sequence}', 'log')
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    train_writer = SummaryWriter(os.path.join(log_savepath, 'train'))
    val_writer = SummaryWriter(os.path.join(log_savepath, 'val'))

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)

    print("len(TrainImgLoader):", len(TrainImgLoader))
    print("len(ValImgLoader):", len(ValImgLoader))

    # loss function choice
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss().to(device)
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'combined':
        loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    else:
        raise ValueError("Unknown Loss Function")

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
                         use_reflectance=_config['use_reflectance'], dropout=_config['dropout'],
                         Action_Func='leakyrelu', attention=False, res_num=18)
    else:
        raise TypeError("Network unknown")
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        saved_state_dict = torch.load(_config['weights'], map_location=device)
        model.load_state_dict(saved_state_dict)
        model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        if starting_epoch != 0:
            starting_epoch = checkpoint['epoch']

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        if _config['optimizer'] != 'adam':
            _run.log_scalar("LR", _config['BASE_LEARNING_RATE'] *
                            math.exp((1 - epoch) * 4e-2), epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = _config['BASE_LEARNING_RATE'] * \
                                    math.exp((1 - epoch) * 4e-2)
        else:
            #scheduler.step(epoch%100)
            _run.log_scalar("LR", scheduler.get_last_lr()[0])


        ## Training ##
        time_for_50ep = time.time()
        for batch_idx, sample in enumerate(TrainImgLoader):
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].to(device)
            sample['rot_error'] = sample['rot_error'].to(device)

            start_preprocess = time.time()
            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose
                real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

                sample['point_cloud'][idx] = sample['point_cloud'][idx].to(device)
                pc_lidar = sample['point_cloud'][idx].clone()

                if _config['max_depth'] < max_depth_behaviour_switch:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

                depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']

                T_mat = tvector2mat(sample['tr_error'][idx])
                R_mat = quat2mat(sample['rot_error'][idx])
                RT = torch.mm(T_mat, R_mat)

                pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc
                #print("pc_rotated:", pc_rotated)

                if _config['max_depth'] < max_depth_behaviour_switch:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

                depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                #print("Initial depth_img:", depth_img)
                #print("Min/max depth_img:", depth_img.min(), depth_img.max())
                #print("")
                #print("Nonzero values:", torch.count_nonzero(depth_img))
                depth_img /= _config['max_depth']
                #print("after first change depth_img:", depth_img)

                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                rgb = sample['rgb'][idx].to(device)
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)


                # print("depth_img:", depth_img)

                # import cv2
                # import random
                # from torchvision.utils import save_image

                # # Normalize to 0–255 for 8-bit grayscale image
                # depth_img_np = depth_img.squeeze().detach().cpu().numpy()
                # depth_img_norm = cv2.normalize(depth_img_np, None, 0, 255, cv2.NORM_MINMAX)
                # depth_img_uint8 = depth_img_norm.astype(np.uint8)

                # # Convert RGB tensor to NumPy image
                # #rgb_np = rgb.detach().cpu().numpy().transpose(1, 2, 0)  # [3, H, W] --> [H, W, 3]
                # #rgb_uint8 = (rgb_np * 255).clip(0, 255).astype(np.uint8)  # assuming rgb is in [0,1]

                # # Write to file
                # filename = f"{random.randint(10**9, 10**10 - 1)}"
                # cv2.imwrite(f"./temp_depth_images/{filename}.png", depth_img_uint8)
                # #cv2.imwrite(f"./temp_depth_images/{filename}_rgb.png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

                # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

                # # move to H×W×3
                # rgb_np = rgb.detach().cpu().numpy().transpose(1, 2, 0)

                # # undo normalization
                # rgb_denorm = (rgb_np * std + mean).clip(0.0, 1.0)

                # save_image(torch.from_numpy(rgb_denorm).permute(2,0,1),
                #         f"./temp_depth_images/{filename}_rgb.png")
                # print(rgb.size())


                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)

            #print("lidar_input list:", lidar_input)
            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input, size=list(input_size), mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=list(input_size), mode="bilinear")
            end_preprocess = time.time()
            #print("\n\n\nHere really the \"lidar_input\":", lidar_input)
            #print("\n\n")
            loss, R_predicted,  T_predicted = train(model, optimizer, rgb_input, lidar_input,
                                                   sample['tr_error'], sample['rot_error'],
                                                   loss_fn, sample['point_cloud'], _config['loss'])

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            if batch_idx % _config['log_frequency'] == 0:
                # show_idx = 0
                # # output image: The overlay image of the input rgb image
                # # and the projected lidar pointcloud depth image
                # rotated_point_cloud = pc_rotated_input[show_idx]
                # R_predicted = quat2mat(R_predicted[show_idx])
                # T_predicted = tvector2mat(T_predicted[show_idx])
                # RT_predicted = torch.mm(T_predicted, R_predicted)
                # rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                # depth_pred, uv = lidar_project_depth(rotated_point_cloud,
                #                                     sample['calib'][show_idx],
                #                                     real_shape_input[show_idx]) # or image_shape
                # depth_pred /= _config['max_depth']
                # depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

                # pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
                # input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
                # gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0))

                # pred_show = torch.from_numpy(pred_show)
                # pred_show = pred_show.permute(2, 0, 1)
                # input_show = torch.from_numpy(input_show)
                # input_show = input_show.permute(2, 0, 1)
                # gt_show = torch.from_numpy(gt_show)
                # gt_show = gt_show.permute(2, 0, 1)

                # train_writer.add_image("input_proj_lidar", input_show, train_iter)
                # train_writer.add_image("gt_proj_lidar", gt_show, train_iter)
                # train_writer.add_image("pred_proj_lidar", pred_show, train_iter)

                train_writer.add_scalar("Loss_Total", loss['total_loss'].item(), train_iter)
                train_writer.add_scalar("Loss_Translation", loss['transl_loss'].item(), train_iter)
                train_writer.add_scalar("Loss_Rotation", loss['rot_loss'].item(), train_iter)
            if _config['loss'] == 'combined':
                train_writer.add_scalar("Loss_Point_clouds", loss['point_clouds_loss'].item(), train_iter)

            local_loss += loss['total_loss'].item()

            if batch_idx % 50 == 0 and batch_idx != 0:

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      #f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
                      f'time for 50 iter: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                _run.log_scalar("Loss", local_loss/50, train_iter)
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset_train), epoch)

        ## Validation ##
        total_val_loss = 0.
        total_val_t = 0.
        total_val_r = 0.

        local_loss = 0.0
        for batch_idx, sample in enumerate(ValImgLoader):
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].to(device)
            sample['rot_error'] = sample['rot_error'].to(device)

            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose
                real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

                sample['point_cloud'][idx] = sample['point_cloud'][idx].to(device)
                pc_lidar = sample['point_cloud'][idx].clone()

                if _config['max_depth'] < max_depth_behaviour_switch:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

                depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']

                reflectance = None
                if _config['use_reflectance']:
                    reflectance = sample['reflectance'][idx].to(device)

                T_mat = tvector2mat(sample['tr_error'][idx])
                R_mat = quat2mat(sample['rot_error'][idx])
                RT = torch.mm(T_mat, R_mat)

                pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc

                if _config['max_depth'] < max_depth_behaviour_switch:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

                depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']

                if _config['use_reflectance']:
                    # This need to be checked
                    # cam_params = sample['calib'][idx].cuda()
                    # cam_model = CameraModel()
                    # cam_model.focal_length = cam_params[:2]
                    # cam_model.principal_point = cam_params[2:]
                    # uv, depth, _, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance)
                    # uv = uv.long()
                    # indexes = depth_img[uv[:,1], uv[:,0]] == depth
                    # refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    # refl_img[uv[indexes, 1], uv[indexes, 0]] = refl[0, indexes]
                    refl_img = None

                # if not _config['use_reflectance']:
                #     depth_img = depth_img.unsqueeze(0)
                # else:
                #     depth_img = torch.stack((depth_img, refl_img))

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

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input, size=input_size, mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=input_size, mode="bilinear")

            loss, transl_e, rot_e, R_predicted,  T_predicted = val(model, rgb_input, lidar_input,
                                                                  sample['tr_error'], sample['rot_error'],
                                                                  loss_fn, sample['point_cloud'], _config['loss'])

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            if batch_idx % _config['log_frequency'] == 0:
                # show_idx = 0
                # # output image: The overlay image of the input rgb image
                # # and the projected lidar pointcloud depth image
                # rotated_point_cloud = pc_rotated_input[show_idx]
                # R_predicted = quat2mat(R_predicted[show_idx])
                # T_predicted = tvector2mat(T_predicted[show_idx])
                # RT_predicted = torch.mm(T_predicted, R_predicted)
                # rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                # depth_pred, uv = lidar_project_depth(rotated_point_cloud,
                #                                     sample['calib'][show_idx],
                #                                     real_shape_input[show_idx]) # or image_shape
                # depth_pred /= _config['max_depth']
                # depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

                # pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
                # input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
                # gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0))

                # pred_show = torch.from_numpy(pred_show)
                # pred_show = pred_show.permute(2, 0, 1)
                # input_show = torch.from_numpy(input_show)
                # input_show = input_show.permute(2, 0, 1)
                # gt_show = torch.from_numpy(gt_show)
                # gt_show = gt_show.permute(2, 0, 1)

                # val_writer.add_image("input_proj_lidar", input_show, val_iter)
                # val_writer.add_image("gt_proj_lidar", gt_show, val_iter)
                # val_writer.add_image("pred_proj_lidar", pred_show, val_iter)

                val_writer.add_scalar("Loss_Total", loss['total_loss'].item(), val_iter)
                val_writer.add_scalar("Loss_Translation", loss['transl_loss'].item(), val_iter)
                val_writer.add_scalar("Loss_Rotation", loss['rot_loss'].item(), val_iter)
            if _config['loss'] == 'combined':
                val_writer.add_scalar("Loss_Point_clouds", loss['point_clouds_loss'].item(), val_iter)

            total_val_t += transl_e
            total_val_r += rot_e
            local_loss += loss['total_loss'].item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('Iter %d val loss = %.3f , time = %.2f' % (batch_idx, local_loss/50.,
                                                                  (time.time() - start_time)/lidar_input.shape[0]))
                local_loss = 0.0
            total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
            val_iter += 1

        print("------------------------------------")
        print('total val loss = %.3f' % (total_val_loss / len(dataset_val)))
        print(f'total translation error: {total_val_t / len(dataset_val)} cm')
        print(f'total rotation error: {total_val_r / len(dataset_val)} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
        _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
        _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)

        # SAVE
        val_loss = total_val_loss / len(dataset_val)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            #_run.result = BEST_VAL_LOSS
            if _config['rescale_transl'] > 0:
                _run.result = total_val_t / len(dataset_val)
            else:
                _run.result = total_val_r / len(dataset_val)
            savefilename = f'{model_savepath}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.pth'

            if hasattr(model, "module"):
                sd = model.module.state_dict()  # multi gpu
            else:
                sd = model.state_dict()  # single gpu

            # only save raw weights
            torch.save(
                sd,
                savefilename
            )

            print(f'Model saved as {savefilename}')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

        # periodic checkpointing every 25 epochs
        if epoch % 25 == 0:
            if hasattr(model, "module"):
                sd = model.module.state_dict()
            else:
                sd = model.state_dict()

            checkpoint_dir = os.path.join(model_savepath, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.pth'
            )
            torch.save(
                sd,
                checkpoint_path
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result
