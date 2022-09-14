bsub -W 24:00 -n 12 -o test_log -R "rusage[mem=20000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=20000]" python test_volsdf_w_bg.py
