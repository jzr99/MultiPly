import os
import shutil
path: str="/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/"
dirs = os.listdir(path)
for dir in dirs:
    if dir == "Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare" or dir == "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2" or dir == "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2" or dir == "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare":
        continue
    sam_mask_path = os.path.join(path, dir, "stage_sam_mask")
    if os.path.exists(sam_mask_path):
        sam_mask_dirs = os.listdir(sam_mask_path)
        for sam_mask_dir in sam_mask_dirs:
            epoch = int(sam_mask_dir)
            sam_mask_path_epoch = os.path.join(sam_mask_path, sam_mask_dir)
            if epoch % 500 != 0 and epoch >= 1:
                # shutil.rmtree(sam_mask_path_epoch)
                for person_i in os.listdir(sam_mask_path_epoch):
                    if person_i != "sam_opt_mask.npy":
                        person_i_path = os.path.join(sam_mask_path_epoch, person_i)
                        shutil.rmtree(person_i_path)
                        print("delete: ", person_i_path)
                # exit()

