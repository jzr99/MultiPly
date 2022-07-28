bsub -W 4:00 -n 12 -G ls_infk -o test_log -R "rusage[mem=40000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=40000]" python test_volsdf_w_bg.py
