#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:15:13 2021

@author: chis
"""
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import superpoint.evaluations.descriptor_evaluation as ev


def draw_matches(data):
    keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
    img1 = np.concatenate([output['image1'], output['image1'], output['image1']], axis=2) * 255
    img2 = np.concatenate([output['image2'], output['image2'], output['image2']], axis=2) * 255
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))


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
    
    #PARSER
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Evaluate detectors of a model.')
    parser.add_argument('--descriptor_name', type=str)
    parser.add_argument('--keep_k_point', type=int, default=1000)
    parser.add_argument('--confidence_threshold', type=int, default=3)
    
    args = parser.parse_args()
    keep_k_points_arg = args.keep_k_point
    confidence_thresholds = [args.confidence_threshold]
    experiments = args.descriptor_name.split()
    
    #VISUALISATION
    # num_images = 1
    # for e in experiments:
    #     orb = True if e[:3] == 'orb' else False
    #     outputs = ev.get_homography_matches(e, keep_k_points=keep_k_points_arg, correctness_thresh=confidence_thresholds, num_images=num_images, orb=orb)
    #     for output in outputs:
    #         img = draw_matches(output) / 255.
    #         plot_imgs([img], titles=[e], dpi=200, cmap='gray')
            
    #DESCRIPTOR EVAL
    for exp in experiments:
        orb = True if exp[:3] == 'orb' else False
        correctness = ev.homography_estimation(exp, keep_k_points=keep_k_points_arg, correctness_thresh=confidence_thresholds, orb=orb)
        print('> {}: {}'.format(exp, correctness))
            
