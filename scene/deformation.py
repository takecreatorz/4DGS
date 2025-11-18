import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.fourier_order_L = args.fourier_order_L
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

        # Head 2 (Residual / DDDM)
        if self.fourier_order_L > 0:
            self.pos_deform_fourier = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3 * self.fourier_order_L * 2))
            self.scales_deform_fourier = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3 * self.fourier_order_L * 2))
            self.rotations_deform_fourier = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4 * self.fourier_order_L * 2))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def calculate_fourier_residual(self, t, f, L):
        # f has shape [N, C, L*2] where C is the dimension of the attribute (e.g., 3 for xyz)
        if L == 0:
            return torch.zeros_like(t.expand(*f.shape[:-1]))

        # Reshape f to [N, C, L, 2] for sin and cos coefficients
        f = f.view(*f.shape[:-1], L, 2)
        f_sin = f[..., 0] # [N, C, L]
        f_cos = f[..., 1] # [N, C, L]

        l_values = torch.arange(1, L + 1, device=t.device).float() # [L]
        t_scaled = 2 * torch.pi * l_values * t.unsqueeze(-1) # [N, 1, L]

        residual = torch.sum(f_sin * torch.sin(t_scaled) + f_cos * torch.cos(t_scaled), dim=-1) # [N, C]
        return residual

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()

        # Head 1 (Base)
        if self.args.no_dx:
            dx_base = torch.zeros_like(rays_pts_emb[:,:3])
        else:
            dx_base = self.pos_deform(hidden)

        if self.args.no_ds :
            ds_base = torch.zeros_like(scales_emb[:,:3])
        else:
            ds_base = self.scales_deform(hidden)
            
        if self.args.no_dr :
            dr_base = torch.zeros_like(rotations_emb[:,:4])
        else:
            dr_base = self.rotations_deform(hidden)

        # Head 2 (Residual / DDDM)
        dx_residual = torch.zeros_like(dx_base)
        ds_residual = torch.zeros_like(ds_base)
        dr_residual = torch.zeros_like(dr_base)
        all_f_coeffs = []

        if self.fourier_order_L > 0:
            if not self.args.no_dx:
                f_x = self.pos_deform_fourier(hidden).view(-1, 3, self.fourier_order_L * 2)
                all_f_coeffs.append(f_x.view(f_x.shape[0], -1))
                dx_residual = self.calculate_fourier_residual(time_emb, f_x, self.fourier_order_L)

            if not self.args.no_ds:
                f_s = self.scales_deform_fourier(hidden).view(-1, 3, self.fourier_order_L * 2)
                all_f_coeffs.append(f_s.view(f_s.shape[0], -1))
                ds_residual = self.calculate_fourier_residual(time_emb, f_s, self.fourier_order_L)

            if not self.args.no_dr:
                f_r = self.rotations_deform_fourier(hidden).view(-1, 4, self.fourier_order_L * 2)
                all_f_coeffs.append(f_r.view(f_r.shape[0], -1))
                dr_residual = self.calculate_fourier_residual(time_emb, f_r, self.fourier_order_L)

        # Final Deformation
        dx = dx_base + dx_residual
        ds = ds_base + ds_residual
        dr = dr_base + dr_residual

        pts = rays_pts_emb[:,:3] * mask + dx
        scales = scales_emb[:,:3] * mask + ds

        if self.args.apply_rotation:
            rotations = batch_quaternion_multiply(rotations_emb, dr)
        else:
            rotations = rotations_emb[:,:4] + dr
        
        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        if all_f_coeffs:
            f_coeffs_cat = torch.cat(all_f_coeffs, dim=1)
        else:
            f_coeffs_cat = None

        return pts, scales, rotations, opacity, shs, f_coeffs_cat
    def get_mlp_parameters(self):
        """Returns parameters of the base MLP, excluding the DDDM heads."""
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name and "fourier" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_dddm_parameters(self):
        """Returns parameters of the new DDDM (Fourier) heads."""
        parameter_list = []
        if self.fourier_order_L > 0:
            for name, param in self.named_parameters():
                if "fourier" in name:
                    parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs, f_coeffs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs, f_coeffs

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters()) 

    def get_dddm_parameters(self):
        return self.deformation_net.get_dddm_parameters()
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb