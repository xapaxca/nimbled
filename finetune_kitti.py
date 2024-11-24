import os
import logging
import time
import argparse
import warnings
import torch
from tqdm import tqdm
from datetime import timedelta
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from networks.resnet_encoder import ResnetEncoder
from networks.monodepth_decoder import MonoDepthDecoderReLU
from networks.swiftdepth_encoder import SwiftFormer_XS, SwiftFormer_S
from networks.swiftdepth_decoder import DepthDecoderReLU
from networks.litemono_encoder import LiteMono
from networks.litemono_decoder import LiteMonoDecoderReLU
from networks.camera_decoder import CameraDecoder
from eval_kitti import eval_kitti
from datasets.kitti.kitti_dataset import KITTIPseudoDepthDataset
from datasets.kitti.kitti_utils import readlines
from loss import CombinedLoss
from utils import *


warnings.filterwarnings("ignore", category=UserWarning)


def finetune(args):
    assert os.path.isdir(args.pretrained_weights), f"Cannot find pretrained_weights at {args.pretrained_weights}"

    depth_encoder_path = os.path.join(args.pretrained_weights, "depth_encoder.pth")
    depth_decoder_path = os.path.join(args.pretrained_weights, "depth_decoder.pth")
    camera_encoder_path = os.path.join(args.pretrained_weights, "camera_encoder.pth")
    camera_decoder_path = os.path.join(args.pretrained_weights, "camera_decoder.pth")

    model_name = args.model_name

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetune_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    project_dir = os.path.join(checkpoint_dir, args.project_name) 
    os.makedirs(project_dir, exist_ok=True)

    model_save_dir = os.path.join(project_dir, 'models') 
    os.makedirs(model_save_dir, exist_ok=True) 

    setup_logging(project_dir)

    # Prepare trained models
    if args.model_name == 'md2_r18':
        depth_encoder = ResnetEncoder(18, False)
        depth_decoder = MonoDepthDecoderReLU(depth_encoder.num_ch_enc)
        scales = [0, 1, 2, 3]
    elif args.model_name == 'md2_r50':
        depth_encoder = ResnetEncoder(50, False)
        depth_decoder = MonoDepthDecoderReLU(depth_encoder.num_ch_enc)
        scales = [0, 1, 2, 3]
    elif args.model_name == 'swiftdepth_s':
        depth_encoder = SwiftFormer_XS()
        depth_decoder = DepthDecoderReLU(num_ch_enc=depth_encoder.embed_dims, enc_name="SwiftFormer_XS")
        scales = [0, 1, 2]
    elif args.model_name == 'swiftdepth':
        depth_encoder = SwiftFormer_S()
        depth_decoder = DepthDecoderReLU(num_ch_enc=depth_encoder.embed_dims, enc_name="SwiftFormer_S")
        scales = [0, 1, 2]
    elif args.model_name == 'litemono_s':
        depth_encoder = LiteMono(model="lite-mono-small")
        depth_decoder = LiteMonoDecoderReLU(depth_encoder.num_ch_enc, scales=range(3))
        scales = [0, 1, 2]
    elif args.model_name == 'litemono':
        depth_encoder = LiteMono(model="lite-mono")
        depth_decoder = LiteMonoDecoderReLU(depth_encoder.num_ch_enc, scales=range(3))
        scales = [0, 1, 2]
    elif args.model_name == 'litemono_8m':
        depth_encoder = LiteMono(model="lite-mono-8m")  
        depth_decoder = LiteMonoDecoderReLU(depth_encoder.num_ch_enc, scales=range(3))
        scales = [0, 1, 2]
    else:
        raise ValueError('Invalid model name')

    depth_encoder = load_weights(depth_encoder, depth_encoder_path)
    depth_decoder = load_weights(depth_decoder, depth_decoder_path)

    camera_encoder = ResnetEncoder(18, True, 2)
    camera_decoder = CameraDecoder(camera_encoder.num_ch_enc[-1], num_ch_dec=256, learn_K=args.learn_k)

    camera_encoder = load_weights(camera_encoder, camera_encoder_path)
    camera_decoder = load_weights(camera_decoder, camera_decoder_path)
    
    # Optimizer
    parameters = []
    parameters.append({'params': depth_encoder.parameters()})
    parameters.append({'params': depth_decoder.parameters()})
    parameters.append({'params': camera_encoder.parameters()})
    parameters.append({'params': camera_decoder.parameters()})
    optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    # Loss function
    loss_fn = {}
    loss_fn["loss"] = CombinedLoss(ssl_mode="min_reconstruction_automasked", lambda_factor=args.lambda_factor)

    # read KITTI
    splits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'kitti', 'splits', 'eigen_zhou')
    fpath = os.path.join(splits_dir, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    img_ext = '.png'
    img_height = 192
    img_width = 640
    frame_ids = [0, -1, 1]
    num_scales = len(scales)

    # DataLoader
    train_dataset = KITTIPseudoDepthDataset(args.data_dir, train_filenames, img_height, img_width, frame_ids, num_scales, is_train=True, img_ext=img_ext)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    depth_encoder.to(device)
    depth_decoder.to(device)
    camera_encoder.to(device)
    camera_decoder.to(device)

    n_params_depth_encoder = sum(p.numel() for p in depth_encoder.parameters())
    n_params_depth_decoder = sum(p.numel() for p in depth_decoder.parameters())
    n_params_camera_encoder = sum(p.numel() for p in camera_encoder.parameters())
    n_params_camera_decoder = sum(p.numel() for p in camera_decoder.parameters())
    
    logging.info(f"Number of parameters of the depth encoder: {n_params_depth_encoder / 1_000_000:.2f} M")
    logging.info(f"Number of parameters of the depth decoder: {n_params_depth_decoder / 1_000_000:.2f} M")
    logging.info(f"Number of parameters of the camera encoder: {n_params_camera_encoder / 1_000_000:.2f} M")
    logging.info(f"Number of parameters of the camera decoder: {n_params_camera_decoder / 1_000_000:.2f} M")

    writer = SummaryWriter(log_dir=os.path.join(project_dir, 'tensorboard_logs'))

    backproj = BackprojectDepth(args.batch_size, img_height, img_width)
    backproj.to(device)

    proj3D = Project3D(args.batch_size, img_height, img_width)
    proj3D.to(device)

    training_start_time = time.time()

    for epoch in range(args.num_epochs):
        depth_encoder.train()
        depth_decoder.train()
        camera_encoder.train()
        camera_decoder.train()

        start_time_epoch = time.time()

        if epoch == args.lambda_step:
            loss_fn["loss"].update_lambda_factor(args.final_lambda_factor)

        for i, batch in tqdm(enumerate(train_loader)):
            start_time_batch = time.time()

            # load dataset
            frame_raw_target = batch["color", 0, 0].to(device)
            frame_aug_target = batch["color_aug", 0, 0].to(device)

            frame_raw_prev = batch["color", -1, 0].to(device)
            frame_aug_prev = batch["color_aug", -1, 0].to(device) 

            frame_raw_next = batch["color", 1, 0].to(device)
            frame_aug_next = batch["color_aug", 1, 0].to(device) 

            pseudo_disp = batch["pseudo_depth"].to(device)

            # camera pose and intrinsics
            camera_net_in_prev2target = torch.cat([frame_aug_prev, frame_aug_target], dim=1)
            camera_net_in_target2next = torch.cat([frame_aug_target, frame_aug_next], dim=1)

            pose_feat_prev2target = camera_encoder(camera_net_in_prev2target)[-1]
            pose_feat_target2next = camera_encoder(camera_net_in_target2next)[-1]

            camera_decoder_out_prev2target = camera_decoder(pose_feat_prev2target) 
            camera_decoder_out_target2next = camera_decoder(pose_feat_target2next) 

            axisangle_prev2target, translation_prev2target = camera_decoder_out_prev2target['R'], camera_decoder_out_prev2target['t']
            axisangle_target2next, translation_target2next = camera_decoder_out_target2next['R'], camera_decoder_out_target2next['t']

            cam_Ts_prev2target = transformation_from_parameters(axisangle_prev2target[:, 0], translation_prev2target[:, 0], invert=True)
            cam_Ts_target2next = transformation_from_parameters(axisangle_target2next[:, 0], translation_target2next[:, 0], invert=False)

            if args.learn_k:
                cam_K = camera_decoder_out_prev2target['K']
                cam_K = resize_K(cam_K, img_height, img_width)
                cam_K_inv = cam_K.inverse()
            else:
                cam_K = batch[("K", 0)].to(device)
                cam_K_inv = batch[("inv_K", 0)].to(device)
            
            # disparities
            feat = depth_encoder(frame_aug_target)
            pred_disps = depth_decoder(feat)
            
            L_psl = 0.0
            L_ssl = 0.0
            L_total = 0.0

            for scale in scales:
                pred_disp = pred_disps[("disp", scale)]

                if scale != 0:
                    pred_disp_out_scale = F.interpolate(pred_disp, 
                                                        [img_height, img_width], 
                                                        mode="bilinear", 
                                                        align_corners=False)
                else:
                    pred_disp_out_scale = pred_disp

                pred_depth = to_inv(pred_disp_out_scale)

                cam_points = backproj(pred_depth, cam_K_inv)
                pix_coords_prev2target = proj3D(cam_points, cam_K, cam_Ts_prev2target)
                pix_coords_target2next = proj3D(cam_points, cam_K, cam_Ts_target2next)

                prev_warped = F.grid_sample(frame_raw_prev, 
                                            pix_coords_prev2target, 
                                            padding_mode="border", 
                                            align_corners=True)
                next_warped = F.grid_sample(frame_raw_next, 
                                            pix_coords_target2next, 
                                            padding_mode="border", 
                                            align_corners=True)

                warped_frames = torch.stack([prev_warped, next_warped])
                support_frames = torch.stack([frame_raw_prev, frame_raw_next])

                losses = loss_fn["loss"](target_frame=frame_raw_target, 
                                         warped_support_frames=warped_frames, 
                                         support_frames=support_frames, 
                                         pred_disp=pred_disp_out_scale, 
                                         pseudo_disp=pseudo_disp)
                
                L_psl += losses["psl"]
                L_ssl += losses["ssl"]
                L_total += losses["total"]
                
            L_psl /= len(scales) 
            L_ssl /= len(scales) 
            L_total /= len(scales) 

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            # log
            if i == 0 or (i + 1) % args.log_every == 0:
                global_step = epoch * len(train_loader) + i

                for j in range(2):
                    img_raw_np = frame_raw_target[j].cpu().detach().numpy().transpose(1, 2, 0)
                    prev_warped_np = prev_warped[j].cpu().detach().numpy().transpose(1, 2, 0)
                    next_warped_np = next_warped[j].cpu().detach().numpy().transpose(1, 2, 0)
                    pred_disp_np = normalize_image(pred_disps[("disp", 0)][j].squeeze()) 
                    pseudo_disp_np = normalize_image(pseudo_disp[j].squeeze()) 

                    writer.add_image(f'{model_name}/frame_{j+1}_01_raw_image', 
                                     img_raw_np, 
                                     global_step=global_step, 
                                     dataformats='HWC')
                    writer.add_image(f'{model_name}/frame_{j+1}_02_prev_warped', 
                                     prev_warped_np, 
                                     global_step=global_step, 
                                     dataformats='HWC')
                    writer.add_image(f'{model_name}/frame_{j+1}_03_next_warped', 
                                     next_warped_np, 
                                     global_step=global_step, 
                                     dataformats='HWC')
                    writer.add_image(f'{model_name}/frame_{j+1}_04_pred_disp', 
                                     pred_disp_np, 
                                     global_step=global_step, 
                                     dataformats='HW')
                    writer.add_image(f'{model_name}/frame_{j+1}_05_pseudo_disp', 
                                     pseudo_disp_np, 
                                     global_step=global_step, 
                                     dataformats='HW')

                writer.add_scalar('Loss/L_total', L_total.item(), global_step=global_step)
                writer.add_scalar('Loss/L_supervised', L_psl.item(), global_step=global_step)
                writer.add_scalar('Loss/L_selfsupervised', L_ssl.item(), global_step=global_step)

                time_elapsed = time.time() - training_start_time
                time_elapsed_hms = str(timedelta(seconds=int(time_elapsed)))
                batch_time = time.time() - start_time_batch
                total_batches_left = (len(train_loader) - (i + 1)) + len(train_loader) * (args.num_epochs - (epoch + 1))
                time_left = batch_time * total_batches_left
                time_left_hms = str(timedelta(seconds=int(time_left)))

                current_lr = scheduler.get_last_lr()[0]

                logging.info(f'Time Elapsed: {time_elapsed_hms}, '
                             f'Time Left: {time_left_hms}, '
                             f'Epoch: [{epoch}/{args.num_epochs - 1}], '
                             f'Step: [{i+1}/{len(train_loader)}], '
                             f'LR: {current_lr}, '
                             f'L_total: {L_total.item():.4f}, '
                             f'L_psl: {L_psl.item():.4f}, '
                             f'L_ssl: {L_ssl.item():.4f}')

        scheduler.step()

        epoch_time = time.time() - start_time_epoch
        epoch_time_hms = str(timedelta(seconds=int(epoch_time)))
        logging.info(f'Epoch [{epoch}] completed in {epoch_time_hms}')

        save_weights(depth_encoder, f'depth_encoder', epoch, model_save_dir)
        save_weights(depth_decoder, f'depth_decoder', epoch, model_save_dir)
        save_weights(camera_encoder, f'camera_encoder', epoch, model_save_dir)
        save_weights(camera_decoder, f'camera_decoder', epoch, model_save_dir)

        logging.info(f'Models saved at the end of epoch {epoch}')

        kitti_results = eval_kitti(data_dir=args.data_dir, 
                                   weights_dir=os.path.join(model_save_dir, f"weights_{epoch}"),
                                   model_name=args.model_name,
                                   eval_split="eigen",
                                   align="median",
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers)
        logging.info(f"Results on KITTI eigen split, median align, after epoch {epoch}:\n{kitti_results}")

    writer.close() 


def main():
    parser = argparse.ArgumentParser(description="Finetune on KITI")
    parser.add_argument("--project_name", 
                        type=str, 
                        required=True,
                        help="Name of the project to be saved")
    parser.add_argument("--model_name", 
                        type=str, 
                        required=True,
                        help="Name of the depth model to finetuned")
    parser.add_argument("--pretrained_weights", 
                        type=str, 
                        required=True,
                        help="Path to the pretrained weights")
    parser.add_argument("--data_dir", 
                        type=str, 
                        required=True,
                        help="Path to the KITTI data directory with .png files")
    parser.add_argument("--lambda_factor", 
                        type=float, 
                        default=0.9, 
                        help="Lambda value for the loss function")
    parser.add_argument("--final_lambda_factor", 
                        type=float, 
                        default=0.95, 
                        help="Lambda value after the step")
    parser.add_argument("--lambda_step", 
                        type=int, 
                        default=15, 
                        help="Lambda step epoch")
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=20, 
                        help='Number of workers')
    parser.add_argument("--learning_rate", 
                        type=float, 
                        default=1e-4, 
                        help="Optimizer learning rate")
    parser.add_argument("--step_size", 
                        type=int, 
                        default=10, 
                        help="Optimizer step epoch")
    parser.add_argument("--step_gamma", 
                        type=float, 
                        default=0.1, 
                        help="Optimizer step multiplier")
    parser.add_argument("--weight_decay", 
                        type=float, 
                        default=1e-3, 
                        help="Optimizer weight decay")
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=16, 
                        help='Batch size')
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=8, 
                        help='Number of workers')
    parser.add_argument("--learn_k",
                        help="If set, learn camera intrinsics",
                        action="store_true")
    parser.add_argument('--log_every', 
                        type=int, 
                        default=200, 
                        help='Tensboard logs every given epoch')
    args = parser.parse_args()
    finetune(args)

if __name__ == '__main__':
    main()
