from typing import ByteString
import numpy as np
import cv2
import os
import argparse
import Utils.MathUtils as mutils
import Utils.ImageUtils as imutils
import Utils.MiscUtils as miscutils


def orb_detect_kpts(image, no_kpts=None):
    orb = cv2.ORB_create(no_kpts)
    kp = orb.detect(image ,None)
    kp, des = orb.compute(image, kp)
    kpt_image = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
    kpt_image = cv2.rotate(kpt_image, cv2.ROTATE_90_CLOCKWISE)
    return kpt_image


def get_orb_matches(image1, image2):
    query_img = image1
    train_img = image2
    
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(queryDescriptors, trainDescriptors)
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Initialize lists
    matches_image1 = list()
    matches_image2 = list()

    for mat in matches:
        query_idx = mat.queryIdx
        train_idx = mat.trainIdx
    
        # Get the coordinates: x - columns, y - rows
        x1, y1 = queryKeypoints[query_idx].pt
        x2, y2 = trainKeypoints[train_idx].pt
        
        matches_image1.append((x1, y1))
        matches_image2.append((x2, y2))
        
    matches_image1 = np.array(matches_image1)
    # print(type(matches_image1[1]))
    matches_image2 = np.array(matches_image2)
    # final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:150], None)
    
    # final_img = cv2.resize(final_img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow("Matches", final_img)
    # cv2.waitKey(0)

    return matches_image1, matches_image2
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputFilePath', default='Data/final_v0.mp4', help='Path to the input file')
    parser.add_argument('-s', '--SaveFileName', default='Output_Files/result.avi', help='Name of the output file')
    
    args = parser.parse_args()
    input_path = args.InputFilePath
    save_path = args.SaveFileName   
    
    # K = calibrateCam(input_path)
    
    cap = cv2.VideoCapture(input_path)
    all_frames = list()
    ret = True
    count = 0
    disp_duration = 3
    while ret:
        ret, frame = cap.read()
        if ret:
            # print(frame.shape)
            all_frames.append(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
            # kpt_frame = orb_detect_kpts(frame, no_kpts=100)
            # cv2.imshow('frame', kpt_frame)
            # if cv2.waitKey(disp_duration) & 0xFF == ord('q'):
                # break
        else:
            break
    
    # frame1, frame2 = all_frames[0], all_frames[1]
    # pt_set1, pt_set2 = get_orb_matches(frame1, frame2)
    # # print(pt_set1, pt_set2)
    # F, best_idxs = mutils.get_inliers(pt_set1, pt_set2, n_iterations=500, error_thresh=0.005)
    # best_pts1, best_pts2 = pt_set1[best_idxs], pt_set2[best_idxs]
    # plt_idxs_pts2 = imutils.get_plot_points(best_pts2, frame1.shape)
    # imutils.plot_matches(frame1, frame2, best_pts1, plt_idxs_pts2)
    
    all_frames = all_frames[:200]

    points_master = list()
    for i in range(0, len(all_frames)-1):
        print(i)
        frame1, frame2 = all_frames[i], all_frames[i+1]
        pt_set1, pt_set2 = get_orb_matches(frame1, frame2)
        # print(pt_set1, pt_set2)
        F, best_idxs = mutils.get_inliers(pt_set1, pt_set2, n_iterations=500, error_thresh=0.005)
        best_pts1, best_pts2 = pt_set1[best_idxs], pt_set2[best_idxs]
        points_master.append(np.hstack((best_pts1, best_pts2)))
    
    array_path = 'Output_Files/matches.npy'
    miscutils.save_np_array(points_master, array_path)
    
    # plt_idxs_pts2 = imutils.get_plot_points(best_pts2, frame1.shape)
    # imutils.plot_matches(frame1, frame2, best_pts1, plt_idxs_pts2)
        

if __name__ == "__main__":
    main()