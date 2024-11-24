import torch
import torch.nn as nn
import numpy as np
import os
import logging
from datetime import datetime


def get_translation_matrix(translation_vector):
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


class BackprojectDepth(nn.Module):
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def build_K(fs, cs):
    b = fs.shape[0]
    K = torch.eye(4, device=fs.device).unsqueeze(0).repeat(b, 1, 1)
    K[:, 0, 0] = fs[:, 0]
    K[:, 1, 1] = fs[:, 1]
    K[:, 0, 2] = cs[:, 0]
    K[:, 1, 2] = cs[:, 1]
    return K


def resize_K(K, height, width):
    K = K.clone()
    K[..., 0, :] *= width
    K[..., 1, :] *= height
    return K


def disp_to_depth(disp, min_depth = 0.01, max_depth = 100.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def normalize_image(x):
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def setup_logging(base_dir):
    logs_dir = os.path.join(base_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'training_{timestamp}.log'
    log_file_path = os.path.join(logs_dir, log_filename)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file_path,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_weights(module, module_name, epoch, checkpoint_dir):
    save_folder = os.path.join(checkpoint_dir, f"weights_{epoch}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    path = os.path.join(save_folder, f"{module_name}.pth")
    module_dict = module.state_dict()
    torch.save(module_dict, path)


def load_weights(model, path):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if 'model_state_dict' in ckpt:
            _state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        state_dict = _state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        print(f"Pretrained weights have been loaded from: {path}")
        print(f"\tTotal number of keys: {len(state_dict.keys())}")
        print(f"\tNumber of missing keys: {len(missing_keys)}")
        print(f"\tNumber of unexpected keys: {len(unexpected_keys)}")
    except Exception:
        print("Pretrained weights could not be loaded")
    return model


def load_optimizer_state(optimizer, path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if 'optimizer_state_dict' in checkpoint:
            state_dict = checkpoint['optimizer_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint  
        optimizer.load_state_dict(state_dict)
        print(f"Optimizer state has been successfully loaded from: {path}")
    except:
        print("Error loading optimizer state")
    return optimizer


def load_weights_camera(model, path):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if 'model_state_dict' in ckpt:
            _state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        state_dict = _state_dict
        state_dict['patch_embed.0.weight'] = torch.cat([state_dict['patch_embed.0.weight']] * 2, 1) / 2
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        print(f"Pretrained weights have been loaded from: {path}")
        print(f"\tTotal number of keys: {len(state_dict.keys())}")
        print(f"\tNumber of missing keys: {len(missing_keys)}")
        print(f"\tNumber of unexpected keys: {len(unexpected_keys)}")
    except:
        print("Pretrained weights could not be loaded")
    return model


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def align_lsqr(pred, target):
    A = np.array([[(pred ** 2).sum(), pred.sum()], [pred.sum(), pred.shape[0]]])
    if np.linalg.det(A) <= 0: return 0, 0

    b = np.array([(pred * target).sum(), target.sum()])
    x = np.linalg.inv(A) @ b
    return x.tolist()


def to_inv(depth, eps=1e-7):
    return (depth > 0) / (depth + eps)


def compute_eigen_metrics(pred, target):
    # Calculate errors
    abs_rel = np.mean(np.abs(pred - target) / target)
    sq_rel = np.mean(((pred - target) ** 2) / target)

    # Calculate RMSE
    rms = np.sqrt(np.mean((pred - target) ** 2))
    rms_log = np.sqrt(np.mean((np.log(pred) - np.log(target)) ** 2))

    # Calculate accuracy under thresholds
    max_ratio = np.maximum(pred / target, target / pred)
    a1 = np.mean(max_ratio < 1.25).astype(float)
    a2 = np.mean(max_ratio < 1.25 ** 2).astype(float)
    a3 = np.mean(max_ratio < 1.25 ** 3).astype(float)

    metrics = np.array([abs_rel, sq_rel, rms, rms_log, a1, a2, a3])

    return metrics


def disp_to_depth(disp, min_depth = 0.01, max_depth = 100.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
