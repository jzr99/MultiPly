from lib.model.idr import IDR
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:03d}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=1,
        save_top_k=-1)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=1500,
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.train.data_dir, 'mean_shape.npy')
    model = IDR(opt, betas_path)
    testset = create_dataset(opt.dataset.test)


    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    trainer.test(model, testset, ckpt_path=checkpoint)



if __name__ == '__main__':
    main()