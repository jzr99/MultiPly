bsub -W 24:00 -n 12 -o test_log -G ls_hilli -R "rusage[mem=20000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=20000]" python test_threedpw.py
