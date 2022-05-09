import numpy as np
import cv2
import os
import argparse


# Getting the world coordinates
def getWorldCorners(edge, n_X, n_Y):
    x_vals = [edge*int(val) for val in np.linspace(0, n_X-1, n_X)]
    y_vals = [edge*int(val) for val in np.linspace(0, n_Y-1, n_Y)]
    w_corners = list()
    for yval in y_vals:
        for xval in x_vals:
            crnr = [[xval, yval, 0]]
            w_corners.append(crnr)    
    w_corners = np.array(w_corners, dtype=np.float32)
    return w_corners


def calibrateCam(image_path):
    frames = list()
    for file_name in sorted(os.listdir(image_path)):
        if ".jpg" in file_name:
            # print(file_name)
            path = os.path.join(image_path, file_name)
            frm = cv2.imread(path)
            h, w = frm.shape[:2]
            # print(h, w)
            # cv2.imshow('frame', frm)
            # cv2.waitKey(0)
            frames.append(frm)
    # cap = cv2.VideoCapture(input_path)
    # count = 0
    # all_frames = list()
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imshow('frame',frame)
    #         all_frames.append(frame)
    #         count += 1
    #         if cv2.waitKey(300) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    # print(count)
    
    # imp_frames = all_frames
    # count = 0
    # frames = list()
    # for i in range(len(imp_frames)):
    #     if count % 1 == 0:
    #         frames.append(imp_frames[i])
    #     count += 1
    # # print(len(frames))
    
    image_points = list()
    world_pts = list()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for frm in frames:
        world_pts.append(getWorldCorners(21.5, 6, 9))
        gray_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_frame, 120, 255, cv2.THRESH_BINARY)
        ret, corners = cv2.findChessboardCorners(thresh, (6, 9), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray_frame, corners, (11,11),(-1,-1), criteria)
            image_points.append(corners2)
            for corner in corners2.reshape(-1, 2):
                corner = (int(corner[0]), int(corner[1]))
                test_frame = cv2.circle(frm, corner, 10, (0, 0, 255), 4)
            test_frame = cv2.resize(test_frame, (0,0), fx=0.2, fy=0.2)
            # cv2.imshow('frame', test_frame)
            # cv2.waitKey(0)
        # frm = cv2.drawChessboardCorners(frm, (6, 9), corners2, ret)

    ret, K_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(world_pts, image_points, gray_frame.shape[::-1], None, None)
    print(K_matrix)
    
    return K_matrix


def get_plot_points(keypoints2, image1_size):
    height, width = image1_size[0], image1_size[1]
    new_keypoints2 = list()
    for point in keypoints2:
        new_width2 = point[0] + width
        new_height2 = point[1]
        new_keypoints2.append((int(new_width2), (int(new_height2))))
    return new_keypoints2


def plot_matches(image1, image2, point_set1, point_set2):
    point_set1 = np.array(point_set1, dtype=int)
    horizontal_concat = np.concatenate((image1, image2), axis=1)
    for i in range(len(point_set1)):
        # print(point_set1[i], point_set2[i])
        horizontal_concat = cv2.line(horizontal_concat, point_set1[i], point_set2[i], (0, 255, 0), 1)
    horizontal_concat = cv2.resize(horizontal_concat, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("concat", horizontal_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def ORBMatcher(image1, image2, no_kpts=None):
    # img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(image1, 150, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image2, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('frame', cv2.resize(thresh1, (0,0), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Creating an Oriented BRIEF object with the desired number of key points
    orb = cv2.ORB_create(no_kpts)   
    kp1 = orb.detect(thresh1, None)
    kp2 = orb.detect(thresh2, None)
    kp1, des1 = orb.compute(image1, kp1)
    kp2, des2 = orb.compute(image2, kp2)
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)  # Mapping matching points
    matches = sorted(matches, key = lambda x:x.distance)
    
    fram = cv2.drawMatches(image1, kp1, image2, kp2, matches, None)
    fram = cv2.resize(fram, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('frame', fram)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    # matches = matcher.match(des1, des2, None)  # Mapping matching points

    # # matches = sorted(matches, key = lambda x:x.distance)
    # # match_lines = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None)
    # # cv2.imshow('Matches', match_lines)

    # # Storing the keypoints into readable arrays
    # unraveled_pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    # unraveled_pts2 = np.zeros((len(matches), 2), dtype=np.float32)
    # for i, match in enumerate(matches):
    #     unraveled_pts1[i, :] = kp1[match.queryIdx].pt   
    #     unraveled_pts2[i, :] = kp2[match.trainIdx].pt

    # plotpts2 = get_plot_points(unraveled_pts2, image1.shape)
    # plot_matches(image1, image2, unraveled_pts1, plotpts2)
    
    
    
    # return unraveled_pts1, unraveled_pts2


def orb_detect_kpts(image, no_kpts=None):
    orb = cv2.ORB_create(no_kpts)
    kp = orb.detect(image ,None)
    kp, des = orb.compute(image, kp)
    kpt_image = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
    kpt_image = cv2.rotate(kpt_image, cv2.ROTATE_90_CLOCKWISE)
    return kpt_image
    
    
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
            all_frames.append(frame)
            # kpt_frame = orb_detect_kpts(frame, no_kpts=100)
            # cv2.imshow('frame', kpt_frame)
            # if cv2.waitKey(disp_duration) & 0xFF == ord('q'):
                # break
        else:
            break
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()