import random
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import requests
from fastai.vision.all import Image, get_files
from tqdm.auto import tqdm


def download_and_unzip(url,
                       data_dir='data',
                       orig_dirname='Magnetic-tile-defect-datasets.-master',
                       new_dirname='MAGNETIC_TILE_SURFACE_DEFECTS'
                       ):
    outfile = Path(data_dir)/'data.zip'
    with open(outfile, 'wb') as out_file:
        content = requests.get(url, stream=True).content
        out_file.write(content)
    zipfile = ZipFile(outfile)
    zipfile.extractall(path=data_dir)
    shutil.move(data_dir/orig_dirname, data_dir/new_dirname)


def dataset_prep(dataset_url,
                 data_dir,
                 orig_dirname,
                 new_dirname):

    dataset_path = data_dir/new_dirname

    if dataset_path.exists() and dataset_path.is_dir():
        shutil.rmtree(dataset_path)

    download_and_unzip(url=dataset_url,
                       data_dir=data_dir,
                       orig_dirname=orig_dirname,
                       new_dirname=new_dirname)

    img_dir_path = dataset_path/'images'
    mask_dir_path = dataset_path/'masks'
    img_dir_path.mkdir(exist_ok=True)
    mask_dir_path.mkdir(exist_ok=True)

    folders = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
    img_fpaths = get_files(dataset_path, folders=folders, extensions='.jpg')
    msk_fpaths = get_files(dataset_path, folders=folders, extensions='.png')

    assert len(img_fpaths) == len(msk_fpaths)

    for p in [img_dir_path, mask_dir_path]:
        p.mkdir(exist_ok=True)

    for img_fpath, msk_fpath in tqdm(zip(img_fpaths, msk_fpaths), total=len(img_fpaths)):
        msk = np.array(Image.open(msk_fpath))
        msk[msk > 0] = 1  # binary segmenation: defect/defect-free
        new_img_fpath = img_dir_path/img_fpath.name
        new_mask_fpath = mask_dir_path/msk_fpath.name
        shutil.copyfile(img_fpath, new_img_fpath)
        Image.fromarray(msk).save(new_mask_fpath)

    assert len(get_files(img_dir_path, extensions='.jpg')) == len(
        get_files(mask_dir_path, extensions='.png'))


def create_test_dataset(img_dir_path,
                        mask_dir_path,
                        train_img_dir_path,
                        train_mask_dir_path,
                        test_img_dir_path,
                        test_mask_dir_path,
                        test_pct,
                        random_state):
    random.seed(random_state)
    img_fpaths = get_files(img_dir_path, extensions='.jpg')
    test_img_fpaths = random.sample(img_fpaths, int(test_pct*len(img_fpaths)))
    train_img_fpaths = [
        fpath for fpath in img_fpaths if fpath not in test_img_fpaths]

    for dir in [train_img_dir_path,
                train_mask_dir_path,
                test_img_dir_path,
                test_mask_dir_path]:
        dir.mkdir(exist_ok=True)

    for img_fpath in test_img_fpaths:
        mask_fpath = mask_dir_path/f'{img_fpath.stem}.png'
        shutil.copy(img_fpath, test_img_dir_path)
        shutil.copy(mask_fpath, test_mask_dir_path)

    for img_fpath in train_img_fpaths:
        mask_fpath = mask_dir_path/f'{img_fpath.stem}.png'
        shutil.copy(img_fpath, train_img_dir_path)
        shutil.copy(mask_fpath, train_mask_dir_path)
