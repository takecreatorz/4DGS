import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# import tinycudann as tcnn
parent_dir = os.path.dirname(os.path.abspath(__file__))


''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        # self.xyz_max = xyz_max
        # self.xyz_min = xyz_min
        # self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        # self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.ones([1, channels, *world_size]))
        # 追加: 潜在コード用の平面を定義
        self.latent_planes = nn.Parameter(torch.randn(1, latent_dim, world_size[0], world_size[1]))
        ###

    def forward(self, xyz, latent_code):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        # if self.channels == 1:
            # out = out.squeeze(-1)
        # 追加: 潜在コード平面からの補間
        latent_code = latent_code.unsqueeze(-1).unsqueeze(-1)  # [batch, latent_dim, 1, 1]
        latent_out = F.grid_sample(self.latent_planes, latent_code, mode='bilinear', align_corners=True)
        latent_out = latent_out.reshape(*shape, self.latent_dim)
        # return out #変更前のリターン
        # 特徴の結合
        return torch.cat([out, latent_out], dim=-1)#変更後のリターン

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.ones([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))
    def set_aabb(self, xyz_max, xyz_min):
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size}'

# class HashHexPlane(nn.Module):
#     def __init__(self,hparams,
#                  desired_resolution=1024,
#                  base_solution=128,
#                  n_levels=4,
#                  ):
#         super(HashHexPlane, self).__init__()

#         per_level_scale = np.exp2(np.log2(desired_resolution / base_solution) / (int(n_levels) - 1))
#         encoding_2d_config = {
#             "otype": "Grid",
#             "type": "Hash",
#             "n_levels": n_levels,
#             "n_features_per_level": 2,
#             "base_resolution": base_solution,
#             "per_level_scale":per_level_scale,
#         }
#         self.xy = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
#         self.yz = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
#         self.xz = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
#         self.xt = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
#         self.yt = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
#         self.zt = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)

#         self.feat_dim = n_levels * 2 *3

#     def forward(self, x, bound):
#         x = (x + bound) / (2 * bound)  # zyq: map to [0, 1]
#         xy_feat = self.xy(x[:, [0, 1]])
#         yz_feat = self.yz(x[:, [0, 2]])
#         xz_feat = self.xz(x[:, [1, 2]])
#         xt_feat = self.xt(x[:, []])
#         return torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)