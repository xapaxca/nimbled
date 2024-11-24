import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_inv


def l1(x, y):
    return torch.abs(y - x).mean(dim=1, keepdim=True)


def ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
    y = F.pad(y, pad=(1, 1, 1, 1), mode='reflect')

    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=0)

    sigma_x  = F.avg_pool2d(x ** 2, kernel_size=3, stride=1, padding=0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, kernel_size=3, stride=1, padding=0) - mu_y ** 2

    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM_out = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    SSIM_out = SSIM_out.mean(dim=1, keepdim=True)

    return SSIM_out


def smoothness_loss(disp, frame):
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)

    grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
    grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(frame[:, :, :, :-1] - frame[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(frame[:, :, :-1, :] - frame[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def photometric_loss(x, y, alpha=0.85):
    photo_loss = alpha * ssim(x, y) + (1 - alpha) * l1(x, y)
    return photo_loss


def reconstruction_loss(target_frame, warped_support_frames, alpha=0.85):
    photo_losses_target_warped = []
    for warped_support_frame in warped_support_frames:
        photo_losses_target_warped.append(photometric_loss(warped_support_frame, target_frame, alpha))

    photo_losses_target_warped = torch.cat(photo_losses_target_warped, dim=1)

    return photo_losses_target_warped.mean()


def min_reconstruction_loss(target_frame, warped_support_frames, alpha=0.85):
    photo_losses_target_warped = []
    for warped_support_frame in warped_support_frames:
        photo_losses_target_warped.append(photometric_loss(warped_support_frame, target_frame, alpha))

    photo_losses_target_warped = torch.cat(photo_losses_target_warped, dim=1)

    min_reconst_loss, _ = torch.min(photo_losses_target_warped, dim=1)

    return min_reconst_loss.mean()


def min_reconstruction_loss_automasked(target_frame, warped_support_frames, support_frames, alpha=0.85):
    photo_losses_target_warped = []
    for warped_support_frame in warped_support_frames:
        photo_losses_target_warped.append(photometric_loss(warped_support_frame, target_frame, alpha))
    
    photo_losses_target_warped = torch.cat(photo_losses_target_warped, dim=1)
    
    photo_losses_target_support = []
    for support_frame in support_frames:
        photo_losses_target_support.append(photometric_loss(support_frame, target_frame, alpha))
    
    photo_losses_target_support = torch.cat(photo_losses_target_support, dim=1)

    photo_losses_target_support += torch.randn_like(photo_losses_target_support) * 1e-5

    combined = torch.cat((photo_losses_target_warped, photo_losses_target_support), dim=1)

    min_reconst_loss_automasked, _ = torch.min(combined, dim=1)

    return min_reconst_loss_automasked.mean()


def normalize_tensor(tensor):
    B, C, H, W = tensor.shape

    flattened_tensor = tensor.reshape(B, -1)
    
    min_vals = torch.min(flattened_tensor, dim=1, keepdim=True)[0]
    max_vals = torch.max(flattened_tensor, dim=1, keepdim=True)[0]

    min_vals = min_vals.reshape(B, 1, 1, 1)
    max_vals = max_vals.reshape(B, 1, 1, 1)

    eps = 1e-6

    ranges = max_vals - min_vals + eps

    normalized_images = (tensor - min_vals) / ranges

    return normalized_images


def median_tensor(tensor):
    reshaped_tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1)
    medians = torch.median(reshaped_tensor, dim=2, keepdim=True).values
    median_tensor = medians.reshape(tensor.size(0), tensor.size(1), 1, 1)
    return median_tensor

    
def ssi_loss(pred, gt):
    normalized_pred = normalize_tensor(pred)
    normalized_gt = normalize_tensor(gt)

    t_d_pred = median_tensor(normalized_pred)
    t_d_gt = median_tensor(normalized_gt)

    s_d_pred = torch.mean(torch.abs(normalized_pred - t_d_pred), dim=(2, 3), keepdim=True)
    s_d_gt = torch.mean(torch.abs(normalized_gt - t_d_gt), dim=(2, 3), keepdim=True)

    scaled_shifted_pred = (normalized_pred - t_d_pred) / s_d_pred
    scaled_shifted_gt = (normalized_gt - t_d_gt) / s_d_gt

    loss = torch.mean(torch.abs(scaled_shifted_pred - scaled_shifted_gt), dim=(2, 3), keepdim=True)

    return loss.mean()


def silog_loss(pred, gt, epsilon=1e-10, variance_factor=0.85, alpha=10.0):
    pred = to_inv(pred)
    gt = to_inv(gt)
    log_diff = torch.log(pred + epsilon) - torch.log(gt + epsilon) 
    loss = torch.sqrt((log_diff ** 2).mean() - variance_factor * (log_diff.mean() ** 2)) * alpha
    return loss


class SmoothnessLoss(nn.Module):
    def __init__(self, lambda_smooth=0.001):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, target_frame, pred_disp):
        loss = {}
        loss["smooth"] = self.lambda_smooth * smoothness_loss(pred_disp, target_frame)
        return loss


class SSLLoss(nn.Module):
    def __init__(self, mode="min_reconstruction_automasked"):
        super().__init__()
        assert mode in ["reconstruction", "min_reconstruction", "min_reconstruction_automasked"]
        self.mode = mode

    def forward(self, target_frame, warped_support_frames, support_frames):
        loss = {}
        if self.mode == "reconstruction":
            loss["ssl"] = reconstruction_loss(target_frame, warped_support_frames)
        elif self.mode == "min_reconstruction":
            loss["ssl"] = min_reconstruction_loss(target_frame, warped_support_frames)
        elif self.mode == "min_reconstruction_automasked":
            loss["ssl"] = min_reconstruction_loss_automasked(target_frame, warped_support_frames, support_frames)
        return loss
    

class PSLLoss(nn.Module):
    def __init__(self, mode="ssi"):
        super().__init__()
        assert mode in ["ssi", "silog"]
        self.mode = mode

    def forward(self,  pred_disp, pseudo_disp):
        loss = {}
        if self.mode == "ssi":
            loss["psl"] = ssi_loss(pred=pred_disp, gt=pseudo_disp)
        elif self.mode == "silog":
            loss["psl"] = silog_loss(pred=pred_disp, gt=pseudo_disp)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, ssl_mode="min_reconstruction_automasked", psl_mode="ssi", lambda_factor=0.9):
        super().__init__()
        self.ssl_mode = ssl_mode
        self.psl_mode = psl_mode

        self.lambda_factor = lambda_factor

        self.ssl_loss = SSLLoss(mode=self.ssl_mode)
        self.psl_loss = PSLLoss(mode=self.psl_mode)

    def forward(self, target_frame, warped_support_frames, support_frames, pred_disp, pseudo_disp):
        loss = {}

        L_psl = self.psl_loss(pred_disp, pseudo_disp)
        loss.update(L_psl)

        L_ssl = self.ssl_loss(target_frame, warped_support_frames, support_frames)
        loss.update(L_ssl)

        total_loss = self.lambda_factor * loss["ssl"] + (1 - self.lambda_factor) * loss["psl"]

        loss["total"] = total_loss
        
        return loss

    def update_lambda_factor(self, new_lambda_factor):
        self.lambda_factor = new_lambda_factor
        print(f"lambda_factor is set to {self.lambda_factor}")
