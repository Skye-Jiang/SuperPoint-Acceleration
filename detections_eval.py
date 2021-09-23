import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from superpoint.settings import EXPER_PATH
import superpoint.evaluations.detector_evaluation as ev


def draw_keypoints(img, corners, color, radius=3, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple(s*np.flip(c, 0)), radius, color, thickness=-1)
    return img
def select_top_k(prob, thresh=0, num=300):    
    pts = np.where(prob > thresh)
    idx = np.argsort(prob[pts])[::-1][:num]
    pts = (pts[0][idx], pts[1][idx])
    return pts

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
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':    
    
    ## PARSER
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Evaluate detectors of a model.')
    parser.add_argument('--detections_name', type=str)
    parser.add_argument('--keep_k_point', type=int, default=300)
    parser.add_argument('--confidence_threshold', type=int, default=0.01)
    parser.add_argument('--distance_thresh', type=int, default=3)
    
    args = parser.parse_args()
    keep_k_points_arg = args.keep_k_point
    confidence_thresholds = [args.confidence_threshold]
    experiments = args.detections_name.split()
    distance_thresh_arg = args.distance_thresh
    
    ## IMAGE VISUALISATION
    for i in range(1):
        for e, thresh in zip(experiments, confidence_thresholds):
            path = Path(EXPER_PATH, "outputs", e, str(i) + ".npz")
            d = np.load(path)
            
            points1 = select_top_k(d['prob'], thresh=thresh)
            im1 = draw_keypoints(d['image'][..., 0] * 255, points1, (0, 255, 0)) / 255.
            
            points2 = select_top_k(d['warped_prob'], thresh=thresh)
            im2 = draw_keypoints(d['warped_image'] * 255, points2, (0, 255, 0)) / 255.
    
            plot_imgs([im1, im2], ylabel=e, dpi=200, cmap='gray',
                      titles=[str(len(points1[0]))+' points', str(len(points2[0]))+' points'])
    ## REPEATABILITY        
    for exp in experiments:
        repeatability = ev.compute_repeatability(exp, keep_k_points=keep_k_points_arg, distance_thresh=distance_thresh_arg, verbose=True)
        print('> {}: {}'.format(exp, repeatability))
