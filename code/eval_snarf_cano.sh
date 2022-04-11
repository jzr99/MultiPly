bsub -W 4:00 -n 12 -o test_log -G ls_hilli -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" python eval_snarf_cano.py
