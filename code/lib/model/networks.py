import torch.nn as nn
import torch
import numpy as np
from .embedders import get_embedder

class ImplicitNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims = [opt.d_in] + list(
            opt.dims) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt

        if opt.multires > 0:
            if opt.embedder_mode == 'hann':
                embed_fn, input_ch = get_embedder(opt.multires, input_dims=opt.d_in, mode=opt.embedder_mode, 
                                                  kick_in_epoch=opt.kick_in_epoch, iter_epoch=0, full_band_epoch=opt.full_band_epoch)
            else:
                embed_fn, input_ch = get_embedder(opt.multires, input_dims=opt.d_in, mode=opt.embedder_mode)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.cond = opt.cond   
        if self.cond == 'smpl':
            self.cond_layer = [0]
            self.cond_dim = 69
        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = 32
        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            # assert(cond=='smpl' or cond == 'anim')
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            
            if self.cond != 'none' and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)
            if opt.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
            if opt.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond, current_epoch=None):
        if input.ndim == 2: input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0: return input

        input = input.reshape(num_batch * num_point, num_dim)
        # input_embed = input_embed if self.embed_fn is None else self.embed_fn(input_embed)

        if self.cond != 'none':
            num_batch, num_cond = cond[self.cond].shape

            input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)

            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embed_fn is not None:
            if self.opt.embedder_mode == 'hann':
                self.embed_fn, _ = get_embedder(self.opt.multires, input_dims=self.opt.d_in, mode=self.opt.embedder_mode,
                                                kick_in_epoch=self.opt.kick_in_epoch, iter_epoch=current_epoch, full_band_epoch=self.opt.full_band_epoch)
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(
            opt.dims) + [opt.d_out]

        self.embedview_fn = None
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

            # embedpos_fn, input_ch = get_embedder(10) # manually set to 10
            # self.embedpos_fn = embedpos_fn
            # dims[0] += (input_ch - 3)
        if self.mode == 'nerf_frame_encoding':
            # self.num_training_frames = opt.num_training_frames
            # self.dim_frame_encoding = 32
            # self.frame_latent_codes = torch.nn.Embedding(self.num_training_frames, self.dim_frame_encoding)  # TODO , sparse=True)
            dims[0] += 32
        if self.mode == 'pose_no_view':
            self.dim_cond_embed = 8
            self.cond_dim = 69
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        
    def forward(self, points, normals, view_dirs, body_pose, feature_vectors, surface_body_parsing, frame_latent_code=None):
        if self.embedview_fn is not None:
            if self.mode == 'nerf_frame_encoding':
                view_dirs = self.embedview_fn(view_dirs)
            elif self.mode == 'pose_no_view':
                points = self.embedview_fn(points)
        # if self.embedpos_fn is not None:
        #     points = self.embedpos_fn(points)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf_frame_encoding':
            # frame_latent_code = self.frame_latent_codes(frame_idx)
            frame_latent_code = frame_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat([view_dirs, frame_latent_code, feature_vectors], dim=-1)
        elif self.mode == 'pose_no_view':
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            rendering_input = torch.cat([points, normals, body_pose, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'pose':
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            if surface_body_parsing is not None:
                # if self.training:
                    # surface_body_parsing = (self.dropout(surface_body_parsing.float()) / 2.).type(torch.LongTensor).to(points.device)
                parsing_feature = self.parsing_embed(surface_body_parsing)
                rendering_input = torch.cat([points, parsing_feature, view_dirs, body_pose, normals, feature_vectors], dim=-1)
            else:
                rendering_input = torch.cat([points, view_dirs, body_pose, normals, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        # x = self.tanh(x)
        x = self.sigmoid(x)
        return x

class ShadowNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims = [opt.d_in] + list(opt.dims) + [opt.d_out]
        dims[0] += 32
        self.num_layers = len(dims)
        self.scaling = 50
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            # torch.nn.init.xavier_uniform_(lin.weight)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.threshold = 0.05
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, points, frame_latent_code):
        frame_latent_code = frame_latent_code.expand(points.shape[0], -1)
        input = torch.cat([points, frame_latent_code], dim=-1)
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x - self.threshold * self.scaling)
        # x = self.relu(x)
        
        return x