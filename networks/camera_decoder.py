import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class CameraDecoder(nn.Module):
    def __init__(self, num_ch_in, num_ch_dec=256, learn_K=False):
        super().__init__()
        self.num_imgs = 2  
        self.num_ch_in = num_ch_in
        self.num_ch_dec = num_ch_dec 

        self.learn_K = learn_K

        self.pose_eps = 0.01 

        self.squeeze = nn.Sequential(
            nn.Conv2d(self.num_ch_in, self.num_ch_dec, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.pose_decoder =nn.Sequential(
            nn.Conv2d(self.num_ch_dec, self.num_ch_dec, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_ch_dec, self.num_ch_dec, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_ch_dec, 6 * self.num_imgs, kernel_size=1),
        )

        if self.learn_K:
            self.focal_decoder =nn.Sequential(
                nn.Conv2d(self.num_ch_dec, self.num_ch_dec, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_ch_dec, self.num_ch_dec, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_ch_dec, 2, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Softplus(),
            )


            self.center_decoder =nn.Sequential(
                nn.Conv2d(self.num_ch_dec, self.num_ch_dec, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_ch_dec, self.num_ch_dec, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_ch_dec, 2, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Sigmoid(),
            )

    # FOR SSL LEARN_K STABILITY ? // NOT NEEDED FOR COMBINED AND PSEUDO

    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def build_K(self, focal, center):
        b = focal.shape[0]  
        K = torch.eye(4, device=focal.device).unsqueeze(0).repeat(b, 1, 1) 
        K[:, 0, 0] = focal[:, 0]  
        K[:, 1, 1] = focal[:, 1]
        K[:, 0, 2] = center[:, 0]  
        K[:, 1, 2] = center[:, 1]
        return K

    def forward(self, x):
        x = self.squeeze(x)
        out_pose = self.pose_decoder(x)
        out_pose = out_pose.mean(3).mean(2)
        out_pose = out_pose.view(-1, 2, 1, 6)
        out_pose = self.pose_eps * out_pose

        R, t = out_pose[..., :3], out_pose[..., 3:] 
        out = {'R': R, 't': t}

        if self.learn_K:
            out_focal = self.focal_decoder(x)
            out_center = self.center_decoder(x)
            K = self.build_K(out_focal, out_center)
            out['K'] = K

        return out




# class CameraNet(nn.Module):
#     def __init__(self, enc_name: str = 'resnet18', pretrained: bool = True, learn_K: bool = True):
#         super().__init__()
#         self.enc_name = enc_name
#         self.pretrained = pretrained
#         self.learn_K = learn_K  # Add learn_K option

#         self.n_imgs = 2  # Number of images used for pose estimation
#         self.encoder = timm.create_model(enc_name, in_chans=3 * self.n_imgs, features_only=True, pretrained=pretrained)
#         self.n_ch_enc = self.encoder.feature_info.channels()
#         self.n_ch_dec = 256  # Number of channels in the decoder layers

#         self.pose_eps = 0.01  # Scaling factor for pose predictions

#         self.squeeze = self.block(self.n_ch_enc[-1], self.n_ch_dec, kernel_size=1)  # Squeeze layer
#         self.decoders = nn.ModuleDict({
#             'pose': self._get_pose_dec(self.n_ch_dec, self.n_imgs)  # Pose decoder
#         })

#         # Conditionally add decoders for focal length and principal point if learn_K is True
#         if self.learn_K:
#             self.decoders['focal'] = self._get_focal_dec(self.n_ch_dec)
#             self.decoders['offset'] = self._get_offset_dec(self.n_ch_dec)

#     @staticmethod
#     def block(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Module:
#         """Defines a basic convolutional block with ReLU activation."""
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
#             nn.ReLU(inplace=True)
#         )

#     def _get_pose_dec(self, n_ch: int, n_imgs: int) -> nn.Sequential:
#         """Defines the pose estimation decoder."""
#         return nn.Sequential(
#             self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
#             self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(n_ch, 6 * n_imgs, kernel_size=1),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Unflatten(dim=-1, unflattened_size=(n_imgs, 6)),
#         )

#     def _get_focal_dec(self, n_ch: int) -> nn.Sequential:
#         """Defines the focal length estimation decoder."""
#         return nn.Sequential(
#             self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
#             self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(n_ch, 2, kernel_size=1),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Softplus(),
#         )

#     def _get_offset_dec(self, n_ch: int) -> nn.Sequential:
#         """Defines the principal point estimation decoder."""
#         return nn.Sequential(
#             self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
#             self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(n_ch, 2, kernel_size=1),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Sigmoid(),
#         )

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """Forward pass for pose and intrinsics estimation."""
#         feat = self.encoder(x)[-1]  # Use the last feature map from the encoder
#         feat = self.squeeze(feat)  # Squeeze the feature map

#         pose_out = self.pose_eps * self.decoders['pose'](feat)
#         R, t = pose_out[..., :3], pose_out[..., 3:]  # Split rotation and translation vectors

#         out = {'R': R, 't': t}
#         # out = R, t

#         if self.learn_K:  # Conditionally compute and return focal lengths and principal points
#             fs = self.decoders['focal'](feat)  # Focal lengths
#             cs = self.decoders['offset'](feat)  # Principal points
#             out['fs'] = fs
#             out['cs'] = cs

        # return out


