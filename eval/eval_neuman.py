import imageio
import torch
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
import lpips
import glob
import os
import cv2

loss_fn_alex = lpips.LPIPS(net='alex')
def eval_metrics(gts, preds):
    results = {
        'ssim': [],
        'psnr': [],
        'lpips': []
    }
    for gt, pred in zip(gts, preds):
        results['ssim'].append(ssim(pred, gt, multichannel=True))
        results['psnr'].append(skimage.metrics.peak_signal_noise_ratio(gt, pred))
        results['lpips'].append(
            float(loss_fn_alex(np_img_to_torch_img(pred[None])/127.5-1, np_img_to_torch_img(gt[None])/127.5-1)[0, 0, 0, 0].data)
        )
    for k, v in results.items():
        results[k] = np.mean(v)
    return results

def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width
    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(np_img, np.ndarray), f'cannot process data type: {type(np_img)}'
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError(f'cannot process this image with shape: {np_img.shape}')

def get_GT_images(seq, gt_alpha_paths):
    if not os.path.exists(f'/home/chen/RGB-PINA/data/{seq}/GT'):
        os.makedirs(f'/home/chen/RGB-PINA/data/{seq}/GT')
    for i in range(len(gt_alpha_paths)):
        gt_img_alpha = cv2.imread(gt_alpha_paths[i], -1)[..., -1] > 127
        gt_img = cv2.imread(os.path.join(f'/home/chen/RGB-PINA/data/{seq}/image', os.path.basename(gt_alpha_paths[i])))
        gt_img = gt_img * gt_img_alpha[..., None] + 255 * (1 - gt_img_alpha[..., None])
        cv2.imwrite(f'/home/chen/RGB-PINA/data/{seq}/GT/%s.png' % os.path.basename(gt_alpha_paths[i]), gt_img)


# results = eval_metrics(gts, neuman_preds)
# import ipdb
# ipdb.set_trace()
if __name__ == '__main__':
    seq = 'jogging'
    gt_alpha_paths = sorted(glob.glob(f'/home/chen/RGB-PINA/data/{seq}/GT_alpha/*.png'))
    get_GT_images(seq, gt_alpha_paths)

    gt_paths = sorted(glob.glob(f'/home/chen/RGB-PINA/data/{seq}/GT/*.png'))
    humannerf_pred_paths = sorted(glob.glob(f'/home/chen/humannerf/experiments/human_nerf/wild/{seq}/single_gpu/test_views_human_w_pose_refine/*.png'))
    neuman_pred_paths = sorted(glob.glob(f'/home/chen/ml-neuman/demo/test_views_human/{seq}/*.png'))
    ours_pred_paths = sorted(glob.glob(f'/home/chen/RGB-PINA/data/{seq}/ours_pred_new/*.png'))
    gts = []
    humannerf_preds = []
    neuman_preds = []
    ours_preds = []
    
    for i in range(len(gt_paths)):
        gts.append(imageio.imread(gt_paths[i]))
        humannerf_preds.append(imageio.imread(humannerf_pred_paths[i]))
        neuman_preds.append(imageio.imread(neuman_pred_paths[i]))
        ours_preds.append(imageio.imread(ours_pred_paths[i]))
    results = eval_metrics(gts, humannerf_preds)
    neuman_results = eval_metrics(gts, neuman_preds)
    ours_results = eval_metrics(gts, ours_preds)
    print('ours_results', ours_results)
    print('neuman_results', neuman_results)
    print('humannerf_results', results)
    
    import ipdb
    ipdb.set_trace()