from lib.model.volsdf_wo_snarf_tcnn import VolSDF
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
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=50,
        save_top_k=-1)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=8000,
        check_val_every_n_epoch=50,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    model = VolSDF(opt)
    trainset = create_dataset(opt.dataset.train)
    validset = create_dataset(opt.dataset.valid)

    if opt.model.is_continue == True:
        checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
        # trainer.validate(model, validset, ckpt_path=checkpoint)
    else: 
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    main()