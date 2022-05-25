import os
import glob
import json
from pathlib import Path
# from fitter.fit_SMPLH_single import SMPLHFitter
from fitter.fit_SMPLH_single_person import SMPLHFitter


def main(args):
    root = Path(args.capture_root)

    scan_path = root / "meshes_vis" 
    scan_paths = sorted(glob.glob(f"{scan_path}/*"))
    pose_paths = root / "pose-estimation" / "keypoints-3d.npy"

    save_path = root / "smpl"

    fitter = SMPLHFitter(args.model_root, debug=False, hands=False)
    fitter.fit(scan_paths, pose_paths, args.gender, save_path)


if __name__ == "__main__":
    import argparse
    from utils.configs import load_config
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('--capture_root', type=str, help='path to the captured data')
    parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
                        help="Path to yml file with config")
    parser.add_argument("--gender", type=Path, default="models")
    args = parser.parse_args()

    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)