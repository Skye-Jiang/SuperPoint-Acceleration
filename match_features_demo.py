import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import glob
# import tensorrt as trt
import tensorflow as tf  # noqa: E402
from tensorflow.python.client import timeline
from superpoint.settings import EXPER_PATH  # noqa: E402
import time
from models import utils
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

def extract_SIFT_keypoints_and_descriptors(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc



def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)#(H=480,W=640)
    img = cv2.resize(img, img_size)#(640,480)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig

def nums(prob,nms,box_nms,top_k):
    if nms:
        if box_nms:
            prob = tf.map_fn(lambda p: utils.box_nms(p, nms, keep_top_k=top_k), prob)
        else:
            prob = tf.map_fn(lambda p: utils.spatial_nms(p, nms), prob)
    return prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('--nms',type=int, default=4,
                        help='The nms should be defined')
    parser.add_argument('--box_nms', type=int, default=1,
                        help='The box_nms should be defined')
    parser.add_argument('--top_k', type=int, default=0,
                        help='The top_k should be defined')
# (480,640)
    parser.add_argument('weights_name', type=str)
    parser.add_argument('img1_path', type=str)
    parser.add_argument('img2_path', type=str)
    parser.add_argument('imgs_path', type=str)
    parser.add_argument('--H', type=int, default=480,
                        help='The height in pixels to resize the images to. \
                                (default: 480)')
    parser.add_argument('--W', type=int, default=640,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()


    weights_name = args.weights_name
    img1_file = args.img1_path
    img2_file = args.img2_path
    imgs_files = args.imgs_path
    # paths = glob.glob(os.path.join(imgs_files, '*.jpg'))
    paths = glob.glob(os.path.join(imgs_files, '*.ppm'))

    img_size = (args.W, args.H)
    keep_k_best = args.k_best

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_tensor = graph.get_tensor_by_name('superpoint/prob:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        # output_prob_nms_tensor = nums(output_prob_tensor,args.nms,args.box_nms,args.top_k)

        img1, img1_orig = preprocess_image(img1_file, img_size)

        st1 = time.time()

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img1, 0)})

#change the output_prob_nms_tensor into output_prob_tensor if not contain the nums.
        for i in range(4):
            out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                                feed_dict={input_img_tensor: np.expand_dims(img1, 0)},
                                options=options, run_metadata=run_metadata)

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_mobilenetV2_01_%d.json' % i, 'w') as f:
                f.write(chrome_trace)

        et1 = time.time()
        print("Time for Out1:", et1 - st1)

        keypoint_map1 = np.squeeze(out1[0])
        descriptor_map1 = np.squeeze(out1[1])
        kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map1, descriptor_map1, keep_k_best)

        img2, img2_orig = preprocess_image(img2_file, img_size)

        # st2 = time.time()
        out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
        # et2 = time.time()
        # print("Time for Out2:", et2 - st2)
        # print("the paths has:", paths, "the length of paths", len(paths))

        keypoint_map2 = np.squeeze(out2[0])
        descriptor_map2 = np.squeeze(out2[1])
        kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map2, descriptor_map2, keep_k_best)

#Get the average time of 11 image
        total = 0
        for path in paths:
            img_i, img_orig_i = preprocess_image(path, img_size)
            st = time.time()
            output = sess.run([output_prob_tensor, output_desc_tensors],
                              feed_dict={input_img_tensor: np.expand_dims(img_i, 0)})
            et = time.time()
            temp_time = et - st
            total += temp_time
        average = total / len(paths)
        print('number of images', len(paths))
        print('the average time=','{:.5f} s'.format(average))


        # Match and get rid of outliers
        m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
        H, inliers = compute_homography(m_kp1, m_kp2)

        # Draw SuperPoint matches
        matches = np.array(matches)[inliers.astype(bool)].tolist()
        matched_img = cv2.drawMatches(img1_orig, kp1, img2_orig, kp2, matches,
                                      None, matchColor=(0, 255, 0),
                                      singlePointColor=(0, 0, 255))
        # print(len(matches))
        cv2.imshow("SuperPoint matches", matched_img)
        cv2.waitKey(0)

        # Compare SIFT matches
        sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(img1_orig)
        sift_kp2, sift_desc2 = extract_SIFT_keypoints_and_descriptors(img2_orig)
        sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
                sift_kp1, sift_desc1, sift_kp2, sift_desc2)
        sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)

        # Draw SIFT matches
        sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
        sift_matched_img = cv2.drawMatches(img1_orig, sift_kp1, img2_orig,
                                           sift_kp2, sift_matches, None,
                                           matchColor=(0, 255, 0),
                                           singlePointColor=(0, 0, 255))
        cv2.imshow("SIFT matches", sift_matched_img)
        # cv2.imwrite("SIFT matches.jpg", sift_matched_img)
        cv2.waitKey(0)