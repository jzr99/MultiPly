import torch.nn as nn
import torch
import numpy as np
import tinycudann as tcnn

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


class ImplicitNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # import ipdb
        # ipdb.set_trace()
        # self.network = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=opt.d_in,
        #     n_output_dims=opt.d_out+opt.feature_vector_size,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": 2.0,
        #     },
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 2,
        #     }
        # )

        self.encoder = tcnn.Encoding(
            n_input_dims=opt.d_in,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 2.0,
            },
            seed=1337
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.n_output_dims, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, opt.d_out + opt.feature_vector_size),
        )

        self.sdf_bounding_sphere = opt.scene_bounding_sphere
        self.sphere_scale = opt.sphere_scale

    def forward(self, input):
        input = (input + 3) / 6
        x = self.encoder(input).to(torch.float32)
        x = self.decoder(x)
        # x = self.network(input)
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        # x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        gradients = None
        # d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        # gradients = torch.autograd.grad(
        #     outputs=sdf,
        #     inputs=x,
        #     grad_outputs=d_output,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]
    
        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

class RenderingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.mode = opt.mode
        self.mode = "tcnn"
        self.embedview_fn = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.network = tcnn.Network(
            n_input_dims=3 + self.embedview_fn.n_output_dims,
            n_output_dims=opt.d_out,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )

        # self.network = nn.Sequential(
        #     nn.Linear(3 + self.embedview_fn.n_output_dims, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, opt.d_out),
        #     nn.Sigmoid(),
        # )

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        elif self.mode == "tcnn":
            rendering_input = torch.cat([points, view_dirs], dim=-1)

        x = self.network(rendering_input)
        return x