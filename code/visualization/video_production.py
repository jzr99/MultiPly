import cv2
import os
import numpy as np
import imageio
import glob

def mask_comparison():
    name = 'video5'
    start_frame = 446
    end_frame = 556

    writer = imageio.get_writer('/home/chen/SelfReconCode/data/Invisible/masks/%s.mp4' % name, fps=30)

    image_dir = '/home/chen/SelfReconCode/data/Invisible/masks'

    files = [f for f in os.listdir(image_dir) if '.png' in f]
    files.sort()

    for f in files[start_frame:end_frame+1]:
        im = os.path.join(image_dir, f)
        writer.append_data(imageio.imread(im))
    writer.close()

    writer = imageio.get_writer('/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_off_weight_2_w_opt_smpl_normal_wo_detach/test_mask/%s.mp4' % name, fps=30)

    image_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_off_weight_2_w_opt_smpl_normal_wo_detach/test_mask'

    files = [f for f in os.listdir(image_dir) if '.png' in f]
    files.sort()

    for f in files[start_frame:end_frame]:
        im = os.path.join(image_dir, f)
        writer.append_data(imageio.imread(im))
    writer.close()

def make_video1():
    seq = 'roger_wo_disp_freeze_20_every_20_opt_pose'
    DIR = '/home/chen/RGB-PINA/code/outputs/ThreeDPW'

    start_frame = 0
    end_frame = 160
    normal_overlay = True
    files = [f for f in os.listdir(os.path.join(DIR, seq, 'test_rendering')) if '.png' in f]
    files.sort()

    writer_rendering = imageio.get_writer(os.path.join(DIR, seq, 'test_rendering', 'test_rendering.mp4'), fps=20)
    writer_normal = imageio.get_writer(os.path.join(DIR, seq, 'test_normal', 'test_normal.mp4'), fps=20)
    writer_fg_rendering = imageio.get_writer(os.path.join(DIR, seq, 'test_fg_rendering', 'test_fg_rendering.mp4'), fps=20)
    if normal_overlay:
        writer_overlay = imageio.get_writer(os.path.join(DIR, seq, 'test_overlay_normal', 'test_overlay_normal.mp4'), fps=20)

    for idx, f in enumerate(files[start_frame:end_frame+1]):
        img_path = os.path.join(DIR, seq, 'test_rendering', f)
        normal_path = os.path.join(DIR, seq, 'test_normal', f)
        fg_path = os.path.join(DIR, seq, 'test_fg_rendering', f)
        writer_rendering.append_data(imageio.imread(img_path))
        writer_normal.append_data(imageio.imread(normal_path))
        writer_fg_rendering.append_data(imageio.imread(fg_path))
        if normal_overlay:
            overlay_path = os.path.join(DIR, seq, 'test_overlay_normal', f)
            writer_overlay.append_data(imageio.imread(overlay_path))
    writer_rendering.close()
    writer_normal.close()
    writer_fg_rendering.close()
    if normal_overlay:
        writer_overlay.close()

def make_video2():
    seq = 'Weipeng_outdoor'
    DIR = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset'

    start_frame = 165
    end_frame = 587
    writer = imageio.get_writer(os.path.join(DIR, seq, 'Mono.mp4'), fps=30)
    for frame in range(start_frame, end_frame+1):
        img_path = os.path.join(DIR, seq, 'Mono', f'frame_{frame:04d}.png')
        writer.append_data(imageio.imread(img_path))
    writer.close()
    

def make_video3():
    seq = 'seattle_wo_disp_freeze_20_every_20'
    DIR = '/home/chen/RGB-PINA/code/outputs/ThreeDPW'

    start_frame = 0
    end_frame = 59
    normal_overlay = False
    files = [f for f in os.listdir(os.path.join(DIR, seq, 'test_canonical_fvr')) if '.png' in f]
    files.sort()

    writer_rendering = imageio.get_writer(os.path.join(DIR, seq, 'test_canonical_fvr', 'test_canonical_fvr.mp4'), fps=10)
    writer_normal = imageio.get_writer(os.path.join(DIR, seq, 'test_canonical_fvr_normal', 'test_canonical_fvr_normal.mp4'), fps=10)
    if normal_overlay:
        writer_overlay = imageio.get_writer(os.path.join(DIR, seq, 'test_overlay_normal', 'test_overlay_normal.mp4'), fps=10)

    for idx, f in enumerate(files[start_frame:end_frame+1]):
        img_path = os.path.join(DIR, seq, 'test_canonical_fvr', f)
        normal_path = os.path.join(DIR, seq, 'test_canonical_fvr_normal', f)

        writer_rendering.append_data(imageio.imread(img_path))
        writer_normal.append_data(imageio.imread(normal_path))
        if normal_overlay:
            overlay_path = os.path.join(DIR, seq, 'test_overlay_normal', f)
            writer_overlay.append_data(imageio.imread(overlay_path))
    writer_rendering.close()
    writer_normal.close()
    if normal_overlay:
        writer_overlay.close()

def make_video4():
    seq = 'Lan_2'
    DIR = f'/home/chen/disk2/RGB-PINA_results/{seq}_wo_disp_freeze_20_every_20_opt_pose'

    image_paths = sorted(glob.glob(f'/home/chen/RGB-PINA/data/{seq}/image/*.png'))
    overlay_normal_paths = sorted(glob.glob(f'/home/chen/disk2/RGB-PINA_results/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_overlay_normal/*.png'))
    image_writer = imageio.get_writer(os.path.join(DIR, 'img.mp4'), fps=30, macro_block_size=1)
    overlay_writer = imageio.get_writer(os.path.join(DIR, 'overlay.mp4'), fps=30, macro_block_size=1)
    # entropy_writer = imageio.get_writer(os.path.join(DIR, seq, 'entropy.mp4'), fps=30)
    for idx, img_path in enumerate(image_paths):
        if idx == 94:
            continue
        overlay_path = overlay_normal_paths[idx]

        img = imageio.imread(img_path)
        overlay = imageio.imread(overlay_path)
        # entropy = imageio.imread(entropy_path)

        image_writer.append_data(img)
        overlay_writer.append_data(overlay)
        # entropy_writer.append_data(entropy)
    image_writer.close()
    overlay_writer.close()
    # entropy_writer.close()
if __name__ == '__main__':
    make_video4()