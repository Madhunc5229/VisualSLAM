import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    matches_image2 = np.array(matches_image2)
    
    return matches_image1, matches_image2
        

def build_projection_matrix(R, C):
    I = np.identity(3)
    IC = np.column_stack((I, -C))
    P = np.dot(R, IC)
    return P


def build_world(F, map_points1, map_points2):
    mtrx = np.trace(np.dot(F, F.T)/2)*np.identity(len(F)) - np.dot(F, F.T)
    h14 = np.sqrt(mtrx[0, 0])
    h24 = mtrx[0, 1] / h14
    h34 = mtrx[0, 2] / h14
    T_config1 = np.array([[0, -h34, h24], [h34, 0, -h14], [-h24, h14, 0]])
    T_config2 = (-1)*np.array(T_config1)
    
    cofactor_F = np.linalg.det(F)*(np.linalg.inv(F).T)
    R_config1 = (cofactor_F.T - np.dot(T_config1, F))*(1/(h14**2 + h24**2 + h34**2))
    R_config2 = (cofactor_F.T - np.dot(T_config2, F))*(1/(h14**2 + h24**2 + h34**2))
    
    RT_combinations = [(T_config1, R_config1), (T_config1, R_config2), (T_config2, R_config1), (T_config2, R_config2)]
    for comb in RT_combinations:
        C_init = np.transpose([0, 0, 0])
        R_init = np.identity(3)
    P1 = build_projection_matrix(R_init, C_init)
    C1 = np.transpose([h14, h24, h34])
    P2 = build_projection_matrix(R_config1, C1)
    # C_possible = []
    # for i in range(2):
        
    # print(best_matches_current)
    # print(P1, P2)
    # X_3D = cv2.triangulatePoints(P1, P2, best_matches1[0], best_matches2[0])
    # print("The triangulated world coords: \n", X_3D)
    # X_3D = list()
    # for i in range(len(best_matches1)):
    #     x_3d = cv2.triangulatePoints(P1, P2, best_matches1[i], best_matches2[i])
    #     x_3d = x_3d / x_3d[-1]
    #     X_3D.append(x_3d[0:3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputFilePath', default='Data/final_v0.mp4', help='Path to the input file')
    parser.add_argument('-l', '--LoadOrNot', default=1, help='1 to load an existing file or 0 to recompute the Fundamental Matrices')
    parser.add_argument('-s', '--SaveFileName', default='Output_Files/result.avi', help='Name of the output file')
    
    args = parser.parse_args()
    input_path = args.InputFilePath
    load_data = bool(int(args.LoadOrNot))
    save_path = args.SaveFileName   
    
    if load_data:
        with open('Output_Files/Frames.npy', 'rb') as f:
            all_frames = np.load(f, allow_pickle=True)
        with open('Output_Files/matches.npy', 'rb') as f:
            all_best_matches = np.load(f, allow_pickle=True)
        with open('Output_Files/fundamental_matrices.npy', 'rb') as f:
            all_F = np.load(f, allow_pickle=True)

        # for i in range(len(all_frames)-1):
        #     frame1, frame2 = all_frames[i], all_frames[i+1]
        #     best_matches1, best_matches2 = np.array_split(all_best_matches[i], 2, axis=1)
        #     # print(best_matches1)
        #     plot_best_matches2 = imutils.get_plot_points(best_matches2, frame1.shape)
        #     imutils.plot_matches(frame1, frame2, best_matches1, plot_best_matches2)
            # funda_matrix = all_F[i]
            # mtrx = np.trace(np.dot(funda_matrix, funda_matrix.T)/2)*np.identity(len(funda_matrix)) - np.dot(funda_matrix, funda_matrix.T)
            # print(mtrx)

        frame1, frame2 = all_frames[0], all_frames[1]
        best_matches1, best_matches2 = np.array_split(all_best_matches[0], 2, axis=1)
        F = all_F[0]
        mtrx = np.trace(np.dot(F, F.T)/2)*np.identity(len(F)) - np.dot(F, F.T)
        # h = np.diag(mtrx)
        # print(mtrx)
        h14 = np.sqrt(mtrx[0, 0])
        h24 = mtrx[0, 1] / h14
        h34 = mtrx[0, 2] / h14
        T_config1 = np.array([[0, -h34, h24], [h34, 0, -h14], [-h24, h14, 0]])
        T_config2 = (-1)*np.array(T_config1)
        
        cofactor_F = np.linalg.det(F)*(np.linalg.inv(F).T)
        R_config1 = (cofactor_F.T - np.dot(T_config1, F))*(1/(h14**2 + h24**2 + h34**2))
        R_config2 = (cofactor_F.T - np.dot(T_config2, F))*(1/(h14**2 + h24**2 + h34**2))
        
        C_init = np.transpose([0, 0, 0])
        R_init = np.identity(3)
        P1 = build_projection_matrix(R_init, C_init)
        C1 = np.transpose([h14, h24, h34])
        P2 = build_projection_matrix(R_config1, C1)
        
        # print(best_matches_current)
        # print(P1, P2)
        # X_3D = cv2.triangulatePoints(P1, P2, best_matches1[0], best_matches2[0])
        # print("The triangulated world coords: \n", X_3D)
        X_3D = list()
        for i in range(len(best_matches1)):
            x_3d = cv2.triangulatePoints(P1, P2, best_matches1[i], best_matches2[i])
            x_3d = x_3d / x_3d[-1]
            X_3D.append(x_3d[0:3])
        # print(X_3D[0])
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for i in range(len(X_3D)):
            ax.scatter3D(X_3D[i][0], X_3D[i][1], X_3D[i][2], color = "green")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("World Coordinates")
        plt.show()
    
    else:    
        cap = cv2.VideoCapture(input_path)
        all_frames = list()
        ret = True
        count = 0
        disp_duration = 3
        while ret:
            ret, frame = cap.read()
            if ret:
                all_frames.append(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
            else:
                break
        
        all_frames = all_frames[:100]

        points_master = list()
        all_F = list()
        for i in range(0, len(all_frames)-1):
            print(i)
            frame1, frame2 = all_frames[i], all_frames[i+1]
            pt_set1, pt_set2 = get_orb_matches(frame1, frame2)
            F, best_idxs = mutils.get_inliers(pt_set1, pt_set2, n_iterations=2000, error_thresh=0.003)
            best_pts1, best_pts2 = pt_set1[best_idxs], pt_set2[best_idxs]
            points_master.append(np.hstack((best_pts1, best_pts2)))
            all_F.append(F)
        
        array_path1 = 'Output_Files/Frames.npy'
        array_path2 = 'Output_Files/matches.npy'
        array_path3 = 'Output_Files/fundamental_matrices.npy'
        miscutils.save_np_array(all_frames, array_path1)
        miscutils.save_np_array(points_master, array_path2)
        miscutils.save_np_array(all_F, array_path3)
            

if __name__ == "__main__":
    main()
