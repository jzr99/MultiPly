from v2a_model_loop import V2AModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lib.datasets.Hi4D import Hi4DTestFreeDataset
from torch.utils.data import DataLoader
import os
import glob

@hydra.main(config_path="confs", config_name="updown_sam_delay_depth_loop_noshare_base")
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
    # testset = create_dataset(opt.dataset.test)
    dataset = Hi4DTestFreeDataset(opt.dataset.test, image_id=8, offset=0, step_list=[42, 324, 318, 0, 36, 330, 336, 342, 348, 354], scale_factor=1.2)
    testset = DataLoader(
            dataset,
            batch_size=opt.dataset.test.batch_size,
            drop_last=opt.dataset.test.drop_last,
            shuffle=opt.dataset.test.shuffle,
            num_workers=opt.dataset.test.worker,
            pin_memory=True,
            persistent_workers=False,
    )


    trainer.test(model, testset, ckpt_path="checkpoints/epoch=1399-loss=0.015640875324606895.ckpt")

if __name__ == '__main__':
    main()