from functools import partial
from pathlib import Path

from dvclive.fastai import DvcLiveCallback
from fastai.data.all import Normalize, RandomSplitter, get_image_files
from fastai.vision.all import (DiceMulti, JaccardCoeff, Resize,
                               SegmentationDataLoaders, aug_transforms,
                               foreground_acc, imagenet_stats, resnet34,
                               unet_learner)


def get_mask_path(img_path, train_mask_dir_path):
    msk_path = train_mask_dir_path/f'{img_path.stem}.png'
    return msk_path


def train_model(train_img_dir_path,
                train_mask_dir_path,
                n_epochs,
                use_cpu,
                lr,
                batch_size,
                model_pickle_path,
                valid_pct,
                img_size,
                seed):
    fpaths = get_image_files(train_img_dir_path)
    item_tfms_sz = img_size
    batch_tfms_sz = img_size
    item_tfms_sz, batch_tfms_sz

    dls = SegmentationDataLoaders.from_label_func(
        path=train_img_dir_path.parent,
        fnames=fpaths,
        label_func=partial(
            get_mask_path, train_mask_dir_path=train_mask_dir_path),
        codes=[0, 1],
        bs=batch_size,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=Resize(item_tfms_sz),
        batch_tfms=[*aug_transforms(size=batch_tfms_sz),
                    Normalize.from_stats(*imagenet_stats)])
    if use_cpu:
        dls.device = 'cpu'
    learn = unet_learner(dls, resnet34, lr=lr, metrics=[foreground_acc,
                         JaccardCoeff, DiceMulti])
    learn.fine_tune(n_epochs, cbs=[DvcLiveCallback()])
    learn.export(fname=Path(model_pickle_path).absolute())
