import torch
import torch.nn.functional as F
# import commentjson as json
from .networks import ImplicitNet
from ..utils.snarf_utils import broyden, hierarchical_softmax
from .smpl import SMPLServer
from pytorch3d import ops
class SMPLDeformer():
    def __init__(self, max_dist=0.05, K=1, gender='male', betas=None):
        super().__init__()

        self.max_dist = max_dist
        self.K = K
        self.smpl = SMPLServer(gender=gender)
        smpl_params_canoical = self.smpl.param_canonical.clone()
        smpl_params_canoical[:, 76:] = torch.tensor(betas).float().to(self.smpl.param_canonical.device)
        cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(smpl_params_canoical, [1, 3, 72, 10], dim=1)
        smpl_output = self.smpl(cano_scale, cano_transl, cano_thetas, cano_betas)
        self.smpl_verts = smpl_output['smpl_verts']
        self.smpl_weights = smpl_output['smpl_weights']
    def forward(self, x, smpl_tfs, return_weights=True, inverse=False, smpl_verts=None):
        if x.shape[0] == 0: return x
        if smpl_verts is None:
            weights = self.query_skinning_weights_smpl(x[None], smpl_verts=self.smpl_verts[0], smpl_weights=self.smpl_weights)
        else:
            weights = self.query_skinning_weights_smpl(x[None], smpl_verts=smpl_verts[0], smpl_weights=self.smpl_weights)
        if return_weights:
            return weights
        # invalid_ids =((skinning_weights[:,-1]==1) )
        # _, smpl_tfs, _, _ = self.smpl(cond['smpl'])
        x_transformed = skinning(x.unsqueeze(0), weights, smpl_tfs, inverse=inverse)

        return x_transformed.squeeze(0)
    def forward_skinning(self, xc, cond, smpl_tfs):
        weights = self.query_skinning_weights_smpl(xc, smpl_verts=self.smpl_verts[0], smpl_weights=self.smpl_weights)
        x_transformed = skinning(xc, weights, smpl_tfs, inverse=False)

        return x_transformed
    def query_skinning_weights_smpl(self, pts, smpl_verts, smpl_weights):

        distance_batch, index_batch, neighbor_points = ops.knn_points(pts, smpl_verts.unsqueeze(0),
                                                                      K=self.K, return_nn=True)

        neighbor_points = neighbor_points[0]
        distance_batch = distance_batch[0]
        index_batch = index_batch[0, :, 0]

        weights = (smpl_weights[:, index_batch, :])
        # static_ids = ((distance_batch>self.max_dist).float().sum(-1)==self.K)
        # weights = torch.cat([weights,torch.zeros(weights.shape[0],1)], axis=-1)
        # weights[static_ids,:] = 0
        # weights[static_ids,-1] = 1

        return weights

    def query_weights(self, xc, cond):
        weights = self.forward(xc, None, return_weights=True, inverse=False)
        return weights

    def forward_skinning_normal(self, xc, normal, cond, tfs, inverse = False):
        # if xc.ndim == 2:
        #     xc = xc.unsqueeze(0)
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc[0], cond)
        # num_batch, num_point, num_dim = normal.shape
        # num_batch, num_point, num_bone = w.shape
        # num_batch, num_bone, num_dim_h, num_dim_h = tfs.shape
        # p = p.reshape(num_batch*num_point, num_dim)
        # w = w.reshape(num_batch*num_point, num_bone)
        # tfs = tfs.reshape(num_batch*num_point, num_dim_h, num_dim_h)
        p_h = F.pad(normal, (0, 1), value=0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            tf_w = torch.einsum('bpn,bnij->bpij', w.double(), tfs.double())
            p_h = torch.einsum('bpij,bpj->bpi', tf_w.inverse(), p_h.double()).float()
        else:
            p_h = torch.einsum('bpn, bnij, bpj->bpi', w.double(), tfs.double(), p_h.double()).float()
        
        return p_h[:, :, :3]
# import tinycudann as tcnn
# with open('config_hash.json') as config_file:
#     config = json.load(config_file)
class ForwardDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """
    def __init__(self, opt, betas, **kwargs):
        super().__init__()

        self.opt = opt
        # self.lbs_network = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=25, encoding_config=config["encoding"], network_config=config["network"]) # ImplicitNet(opt)
        self.lbs_network = ImplicitNet(opt)
        self.soft_blend = 20
        self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19]
        self.smpl_deformer = SMPLDeformer(betas=betas)
    def forward(self, xd, cond, tfs, eval_mode=False):
        """Given deformed point return its caonical correspondence
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
        # import pdb
        # pdb.set_trace()
        if xd.ndim == 2:
            xd = xd.unsqueeze(0)
        xc_init = self.init(xd, tfs)

        xc_opt, others = self.search(xd,
                                     xc_init,
                                     cond,
                                     tfs,
                                     eval_mode=eval_mode)

        if eval_mode:
            return xc_opt, others

        # compute correction term for implicit differentiation during training

        # do not back-prop through broyden
        xc_opt = xc_opt.detach()

        # reshape to [B,?,D] for network query
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_opt = xc_opt.reshape((n_batch, n_point * n_init, n_dim))

        xd_opt = self.forward_skinning(xc_opt, cond, tfs)

        grad_inv = self.gradient(xc_opt, cond, tfs).inverse()

        correction = xd_opt - xd_opt.detach()
        correction = torch.einsum("bnij,bnj->bni", -grad_inv.detach(),
                                  correction)

        # trick for implicit diff with autodiff:
        # xc = xc_opt + 0 and xc' = correction'
        xc = xc_opt + correction

        # reshape back to [B,N,I,D]
        xc = xc.reshape(xc_init.shape)

        return xc, others

    def init(self, xd, tfs):
        """Transform xd to canonical space for initialization
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc_init (tensor): gradients. shape: [B, N, I, D]
        """
        n_batch, n_point, _ = xd.shape
        _, n_joint, _, _ = tfs.shape

        xc_init = []
        for i in self.init_bones:
            w = torch.zeros((n_batch, n_point, n_joint), device=xd.device)
            w[:, :, i] = 1
            # import pdb
            # pdb.set_trace()
            xc_init.append(skinning(xd, w, tfs, inverse=True))

        xc_init = torch.stack(xc_init, dim=2)

        return xc_init

    def search(self, xd, xc_init, cond, tfs, eval_mode=False):
        """Search correspondences.
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """
        # reshape to [B,?,D] for other functions
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_init = xc_init.reshape(n_batch, n_point * n_init, n_dim)
        xd_tgt = xd.repeat_interleave(n_init, dim=1)

        # compute init jacobians
        if not eval_mode:
            J_inv_init = self.gradient(xc_init, cond, tfs).inverse()
        else:
            w = self.query_weights(xc_init, cond, mask=None)
            J_inv_init = torch.einsum("bpn,bnij->bpij", w,
                                      tfs)[:, :, :3, :3].inverse()

        # reshape init to [?,D,...] for boryden
        xc_init = xc_init.reshape(-1, n_dim, 1)
        J_inv_init = J_inv_init.flatten(0, 1)

        # construct function for root finding
        def _func(xc_opt, mask=None):
            # reshape to [B,?,D] for other functions
            xc_opt = xc_opt.reshape(n_batch, n_point * n_init, n_dim)
            xd_opt = self.forward_skinning(xc_opt, cond, tfs, mask=mask)
            error = xd_opt - xd_tgt
            # reshape to [?,D,1] for boryden
            error = error.flatten(0, 1)[mask].unsqueeze(-1)
            return error

        # run broyden without grad
        with torch.no_grad():
            result = broyden(_func, xc_init, J_inv_init)

        # reshape back to [B,N,I,D]
        xc_opt = result["result"].reshape(n_batch, n_point, n_init, n_dim)
        result["valid_ids"] = result["valid_ids"].reshape(
            n_batch, n_point, n_init)

        return xc_opt, result

    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point
        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        w = self.query_weights(xc, cond, mask=mask)
        xd = skinning(xc, w, tfs, inverse=False)
        return xd

    def backward_skinning(self, xd, cond, tfs, mask=None):
        w = self.query_weights(xd, cond, mask=mask)
        xc = skinning(xd, w, tfs, inverse=True)
        return xc
    def forward_skinning_normal(self, xc, normal, cond, tfs, mask=None, inverse = False):
        if xc.ndim == 2:
            xc = xc.unsqueeze(0)
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc, cond, mask=mask)
        # num_batch, num_point, num_dim = normal.shape
        # num_batch, num_point, num_bone = w.shape
        # num_batch, num_bone, num_dim_h, num_dim_h = tfs.shape
        # p = p.reshape(num_batch*num_point, num_dim)
        # w = w.reshape(num_batch*num_point, num_bone)
        # tfs = tfs.reshape(num_batch*num_point, num_dim_h, num_dim_h)
        p_h = F.pad(normal, (0, 1), value=0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            tf_w = torch.einsum('bpn,bnij->bpij', w.double(), tfs.double())
            p_h = torch.einsum('bpij,bpj->bpi', tf_w.inverse(), p_h.double()).float()
        else:
            p_h = torch.einsum('bpn, bnij, bpj->bpi', w.double(), tfs.double(), p_h.double()).float()
        
        return p_h[:, :, :3]

    def query_weights(self, xc, cond, mask=None):
        """Get skinning weights in canonical space
        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): valid indices. shape: [B, N]
        Returns:
            w (tensor): skinning weights. shape: [B, N, J]
        """
        # w = self.lbs_network(xc[0])[None] 
        w = self.lbs_network(xc, cond)
        w = self.soft_blend * w

        if self.opt.softmax_mode == "hierarchical":
            w = hierarchical_softmax(w)
        else:
            w = F.softmax(w, dim=-1)
        # w = self.smpl_deformer.query_skinning_weights_smpl(xc, self.smpl_deformer.smpl_verts[0], self.smpl_deformer.smpl_weights)
        return w

    def query_smpl_weights(self, xc):

        w = self.smpl_deformer.query_skinning_weights_smpl(xc, self.smpl_deformer.smpl_verts[0], self.smpl_deformer.smpl_weights)
        return w
    def gradient(self, xc, cond, tfs):
        """Get gradients df/dx
        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            grad (tensor): gradients. shape: [B, N, D, D]
        """
        xc.requires_grad_(True)

        xd = self.forward_skinning(xc, cond, tfs)

        grads = []
        for i in range(xd.shape[-1]):
            d_out = torch.zeros_like(xd, requires_grad=False, device=xd.device)
            d_out[:, :, i] = 1
            grad = torch.autograd.grad(
                outputs=xd,
                inputs=xc,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad)

        return torch.stack(grads, dim=-2)


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
    return x_h[:, :, :3]