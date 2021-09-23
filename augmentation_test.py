import cv2
import numpy as np
from superpoint.datasets.coco import Coco
from superpoint.datasets.patches_dataset import PatchesDataset
import matplotlib.pyplot as plt
import tensorflow as tf
def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        traverse = False
        info = imgs[i].shape
        height = info[0]
        weight = info[1]
        for h in range(0,height):
            for w in range(0,weight):
                (r,g,b) =imgs[i][h,w] * 255
                r = int(r)
                g = int(g)
                b = int(b)
                if (((r,g,b) != (255, 0, 0))and((r,g,b)!=(255, 255, 255))and((r,g,b)!=(0, 0, 255)))and(traverse):
                # if h == 191 and w == 121 and traverse:
                    print('H:',h)
                    print('W:',w)
                    print("(r,g,b):",(r,g,b))
        #print(imgs[0][190,60] * 255)
        #[251.00002 254.00002 255.     ]
        cv2.imshow(str(i) + '.jpg', imgs[i])
        cv2.waitKey(0)
        #cv2.imwrite('/home/shuming/5.20/origin.jpg', imgs[0]*255)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)

    plt.tight_layout()

config = {
    'labels': "/home/shared_data/chenyu_eventcamera/exper_darkpoint/outputs/magic-point_coco-export1",

    'augmentation' : {
        'warped_pair': {
            'enable': True,
            'params': {
                'translation': True,
                'rotation': True,
                'scaling': True,
                'perspective': True,
                'scaling_amplitude': 0.2,
                'perspective_amplitude_x': 0.2,
                'perspective_amplitude_y': 0.2,
                'patch_ratio': 0.85,
                'max_angle': 1.57,
                'allow_artifacts': True,
                }
            },
        'photometric': {
            'enable': True,
            'primitives': [
                'random_brightness',
                'random_contrast',
                'additive_speckle_noise',
                'additive_gaussian_noise',
                'additive_shade',
                'motion_blur'
            ],
            'params': {
                'random_brightness': {'max_abs_change': 50},
                'random_contrast': {'strength_range': [0.5, 1.5]},
                'additive_gaussian_noise': {'stddev_range': [0, 10]},
                'additive_speckle_noise': {'prob_range': [0, 0.0035]},
                'salt_pepper_noise':{'salt_vs_pepper':0.2,'amount':0.004},
                'additive_shade': {'transparency_range': [-.5, .5], 'kernel_size_range': [100, 150]},
                'motion_blur': {'max_kernel_size': 3},
            }
        },
        'homographic': {
            'enable': False,
            'params': {
                'translation': True,
                'rotation': True,
                'scaling': True,
                'perspective': True,
                'scaling_amplitude': 0.2,
                'perspective_amplitude_x': 0.2,
                'perspective_amplitude_y': 0.2,
                'patch_ratio': 0.85,
                'max_angle': 1.57,
                'allow_artifacts': True,
            },
            'valid_border_margin': 3,
        }
    }
}
dataset = Coco(**config)
data = dataset.get_training_set()
add_keypoints = False


def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple(s*np.flip(c, 0)), radius, color, thickness=-1)
    return img
def draw_overlay(img, mask, color=[0, 0, 255], op=0.5, s=3):
    mask = cv2.resize(mask.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    img[np.where(mask)] = img[np.where(mask)]*(1-op) + np.array(color)*op
def display(d):
    img = draw_keypoints(d['image'][..., 0] * 255, np.where(d['keypoint_map1']), (0, 255, 0)) if add_keypoints \
           else d['image']

    #draw_overlay(img, np.logical_not(d['valid_mask']))
    return img


for i in range(1):
    images, names = [], []

    for _ in range(1):
        d = next(data)
        images.append(display(d))
        names.append(d['name'])
    plot_imgs(images, titles=names, dpi=200,cmap='brg')
