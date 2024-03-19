from v2a_model_loop import V2AModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob

@hydra.main(config_path="confs", config_name="Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_noshare_base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=10,
        save_top_k=-1)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        devices=1,
        # gpus=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=10000,
        check_val_every_n_epoch=10,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.train.data_dir, 'mean_shape.npy')
    model = V2AModel(opt, betas_path)
    # checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    testset = create_dataset(opt.dataset.test)
    trainer.test(model, testset, ckpt_path="checkpoints/epoch=5549-loss=0.009949694387614727.ckpt")
    # trainer.test(model, testset, ckpt_path="checkpoints/epoch=1049-loss=0.015038730576634407.ckpt")

if __name__ == '__main__':
    main()