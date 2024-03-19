import cv2
import os
import glob
# pred_v2a_dir = '/media/ubuntu/hdd/easymocap/EasyMocap/neuralbody/V2A_output/pair16/test_fg_rendering/-1'
# pred_v2a_dir = "/media/ubuntu/hdd/V2A_output/Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_edge_noshare/test_fg_rendering_88/-1"
# pred_v2a_dir = "/media/ubuntu/hdd/V2A_output/Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare/test_fg_rendering_88/-1"
# pred_v2a_dir = "/media/ubuntu/hdd/V2A_output/Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2_noshare/test_fg_rendering_16/-1"
pred_v2a_dir = "/media/ubuntu/hdd/V2A_output/Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare_sam/test_fg_rendering_88/-1"
pred_body_dir = '/media/ubuntu/hdd/easymocap/EasyMocap/neuralbody/pair19_piggyback19_vitpose_4/test_400/'
save_dir = '/media/ubuntu/hdd/easymocap/EasyMocap/neuralbody/pair19_piggyback19_vitpose_4/gt_gtpose/'
gt_dir = '/media/ubuntu/hdd/Hi4D/pair19/piggyback19'
gt_img_dir = os.path.join(gt_dir, 'images', '88')
gt_mask_dir = os.path.join(gt_dir, 'seg', 'img_seg_mask', '88', 'all')
gt_img_paths = sorted(glob.glob(f'{gt_img_dir}/*.jpg'))
gt_mask_paths = sorted(glob.glob(f'{gt_mask_dir}/*.png'))
pred_v2a_list = sorted(glob.glob(f'{pred_v2a_dir}/*.png'))
pred_body_list = sorted(glob.glob(f'{pred_body_dir}/rgb_map_*.jpg'))
scale_factor = 2

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
        # import pdb;pdb.set_trace()
        results['ssim'].append(ssim(pred, gt, multichannel=True, channel_axis=2))
        results['psnr'].append(skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=255))
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

# def get_GT_images(seq, gt_alpha_paths):
#     if not os.path.exists(f'/home/chen/RGB-PINA/data/{seq}/GT'):
#         os.makedirs(f'/home/chen/RGB-PINA/data/{seq}/GT')
#     for i in range(len(gt_alpha_paths)):
#         gt_img_alpha = cv2.imread(gt_alpha_paths[i], -1)[..., -1] > 127
#         gt_img = cv2.imread(os.path.join(f'/home/chen/RGB-PINA/data/{seq}/image', os.path.basename(gt_alpha_paths[i])))
#         gt_img = gt_img * gt_img_alpha[..., None] + 255 * (1 - gt_img_alpha[..., None])
#         cv2.imwrite(f'/home/chen/RGB-PINA/data/{seq}/GT/%s.png' % os.path.basename(gt_alpha_paths[i]), gt_img)


# results = eval_metrics(gts, neuman_preds)
# import ipdb
# ipdb.set_trace()
# if __name__ == '__main__':
#     seq = 'parkinglot'
#     gt_alpha_paths = sorted(glob.glob(f'/home/chen/RGB-PINA/data/{seq}/GT_alpha/*.png'))
#     get_GT_images(seq, gt_alpha_paths)

#     gt_paths = sorted(glob.glob(f'/home/chen/RGB-PINA/data/{seq}/GT/*.png'))
#     humannerf_pred_paths = sorted(glob.glob(f'/home/chen/humannerf/experiments/human_nerf/wild/{seq}/single_gpu/test_views_human_w_pose_refine/*.png'))
#     neuman_pred_paths = sorted(glob.glob(f'/home/chen/ml-neuman/demo/test_views_human/{seq}/*.png'))
#     ours_pred_paths = sorted(glob.glob(f'/home/chen/RGB-PINA/data/{seq}/ours_pred_new_new/*.png'))
#     gts = []
#     humannerf_preds = []
#     neuman_preds = []
#     ours_preds = []
    
#     for i in range(len(gt_paths)):
#         gts.append(imageio.imread(gt_paths[i]))
#         humannerf_preds.append(imageio.imread(humannerf_pred_paths[i]))
#         neuman_preds.append(imageio.imread(neuman_pred_paths[i]))
#         ours_preds.append(imageio.imread(ours_pred_paths[i]))
#     results = eval_metrics(gts, humannerf_preds)
#     neuman_results = eval_metrics(gts, neuman_preds)
#     ours_results = eval_metrics(gts, ours_preds)
#     print('ours_results', ours_results)
#     print('neuman_results', neuman_results)
#     print('humannerf_results', results)
    
#     import ipdb
#     ipdb.set_trace()







gts = []
body_preds = []
ours_preds = []

for idx, (gt, gt_mask, v2a, body) in enumerate(zip(gt_img_paths, gt_mask_paths, pred_v2a_list, pred_body_list)):
    gt = cv2.imread(gt)
    gt_mask = cv2.imread(gt_mask)
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    v2a = cv2.imread(v2a)
    body = cv2.imread(body)
    gt_masked = cv2.bitwise_and(gt, gt, mask=gt_mask)
    # import pdb;pdb.set_trace()
    # change the masked background to white
    gt_masked[gt_masked == 0] = 255
    gt_masked = cv2.resize(gt_masked, (gt_masked.shape[1] // scale_factor, gt_masked.shape[0] // scale_factor))
    
    gts.append(np.array(gt_masked))
    body_preds.append(np.array(body))
    ours_preds.append(np.array(v2a))
    # for visualization
    os.makedirs(os.path.join(save_dir, 'image'), exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), gt_masked)
    # write label on the bottom of each images
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gt_masked, 'Reference', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(v2a, 'Ours', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(body, 'L-NeuralBody', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # horizontal concat gt_masked and v2a and body
    concat_img = cv2.hconcat([gt_masked, body, v2a])
    os.makedirs(os.path.join(save_dir, 'concat'), exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'concat/%04d.png' % idx), concat_img)

body_results = eval_metrics(gts, body_preds)
ours_results = eval_metrics(gts, ours_preds)
print('body_results', body_results)
print('ours_results', ours_results)










