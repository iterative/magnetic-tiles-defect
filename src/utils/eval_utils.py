import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import (PILImage, get_image_files, load_learner)
from PIL import Image
from tqdm.auto import tqdm
from fastai.data.all import get_image_files


def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    if (np.sum(y_true) == 0) and (np.sum(y_pred) == 0):
        return 1
    else:
        return (2*intersection) / (np.sum(y_true) + np.sum(y_pred))


def resize_and_crop_center(im, img_size=128):
    new_shape = (np.array(im.shape)/np.min(im.shape)*img_size).astype(int)
    im = im.resize((new_shape[1], new_shape[0]))
    # Crop the center of the image
    h, w = im.shape
    left = (w - img_size)/2
    top = (h - img_size)/2
    right = (w + img_size)/2
    bottom = (h + img_size)/2
    im = im.crop((left, top, right, bottom))
    return im


def get_metrics(test_img_dir_path,
                test_mask_dir_path,
                model_pickle_fpath,
                test_img_out_dir,
                img_size,
                save_test_preds
                ):
    learn = load_learner(model_pickle_fpath, cpu=True)   
    test_images = get_image_files(test_img_dir_path)
    inter = 0
    union = 0
    acc = 0
    for img_path in tqdm(test_images):
        mask_path = test_mask_dir_path/f'{img_path.stem}.png'
        im = Image.open(img_path)
        mask_im = Image.open(mask_path)
        im_resized = resize_and_crop_center(im, img_size=img_size)
        y_true = resize_and_crop_center(mask_im, img_size=img_size)
        y_true = np.array(y_true).astype(int)
        y_pred, *_ = learn.predict(PILImage(im_resized))
        y_pred = np.array(y_pred).astype(int)
        acc += (y_true == y_pred).mean()

        inter += (y_pred*y_true).sum()
        union += (y_pred+y_true).sum()
        if save_test_preds:
            fig, axarr = plt.subplots(1, 3)
            fig.suptitle(
                f'{img_path.stem} Image/True/Pred')
            axarr[0].imshow(im_resized, cmap='gray')
            axarr[1].imshow(y_true, cmap='gray')
            axarr[2].imshow(y_pred, cmap='gray')
            plt.savefig(test_img_out_dir/f'{img_path.stem}.jpg', dpi=70)

    dice_mean = 2. * inter/union if union > 0 else None
    jacc_mean = inter/(union-inter) if union > 0 else None
    acc_mean = acc/len(test_images)
    return {'dice_mean': dice_mean, 
            'jacc_mean': jacc_mean, 
            'acc_mean': acc_mean}
