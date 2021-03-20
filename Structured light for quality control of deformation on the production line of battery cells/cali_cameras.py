import glob

import cv2
import numpy as np


# 标定相机，并且储存管理各种参数
#  参数的存储，再确认一下，用哪种方式好。
class Cali_Camera:

    def __init__(self):
        """
        initial all parameter for camera calibration
        """
        # 内参
        self.IntrinsicMatrix = None
        self.RotationMatrices = None
        self.TranslationVectors = None
        self.distortionParameter = None
        # parameter of chessboard
        self.cbrow = 9
        self.cbcol = 5
        self.cbsize = 25
        self.accuracy = 0.001

    def save_parameter(self,path_camera = "parameter_Intrinsic.npz"):
        """
        save the camera parameter
        :return: None
        """
        np.savez(path_camera, IntrinsicMatrix=self.IntrinsicMatrix,
                 RotationMatrices=self.RotationMatrices,
                 TranslationVectors=self.TranslationVectors, distortionParameter=self.distortionParameter)
        print("\n################################")
        print("##Parmeter of camera is saved.##")
        print("################################")

    def load_parameter(self, fname):
        """
        loading the parameter of the camera from npz file
        :param fname: the file path of the camera parameters
        :return: None
        """
        parameter_Intrinsic = np.load(fname)

        self.IntrinsicMatrix = parameter_Intrinsic['IntrinsicMatrix']
        self.RotationMatrices = parameter_Intrinsic['RotationMatrices']
        self.TranslationVectors = parameter_Intrinsic['TranslationVectors']
        self.distortionParameter = parameter_Intrinsic['distortionParameter']


        print("\n##############################################")
        print("##Parmeter of camera is loaded for the file.##")
        print("##############################################")

    def cali_camera_python(self, pict_add, cbrow_in , cbcol_in, cbsize_in, accuracy_in):
        """

        :param pict_add: the image path of calibration picture
        :param cbrow_in: the input parameter for corner point number in a row
        :param cbcol_in: the input parameter for corner point number in a columns
        :param cbsize_in: the input parameter for grid size
        :param accuracy_in: the input parameter for extraction parameter
        :return: None
        """

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cbsize_in, accuracy_in)
        cbrow = cbrow_in
        cbcol = cbcol_in
        self.cbrow = cbrow_in
        self.cbcol = cbcol_in
        self.cbsize = cbsize_in
        self.accuracy = accuracy_in
        # 获取标定板角点的位置
        objp = np.zeros((cbcol * cbrow, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2) # The world coordinate system is built on the calibration plate, and all points have zero z-coordinates, so only x and y need to be assigned


        obj_points = []  # save 3D point coordinate
        img_points = []  # save 2D point coordinate

        images = glob.glob(pict_add)
        i = 0
        print("show the calibration images as follow:")
        print(images)
        for fname in images:
            img = cv2.imread(fname)

            # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) # if the size of the pict is too big, this command helps to speed up the program.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('img',img)
            # cv2.waitKey(1000)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)

            # # draw the corner point extraction on the image and show the result on scream
            # print(corners)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            criteria)  # get the coordinate in subpixels of the corner points
                img_points.append(corners2)

                cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)  # 记住，OpenCV的绘制函数一般无返回值
                cv2.imshow('img', img)
                i += 1;
                # cv2.imwrite('conimg'+str(i)+'.jpg', img)
                cv2.waitKey(100)

        print(f"the total number of the picture ist {len(img_points)}")
        cv2.destroyAllWindows()

        # start calibration of cameras
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) # if resize is done previously,this should also be done

        print("intrinsic matrix of camera:\n", mtx)  # intrinsic parameters
        print("distortion parameter:\n", dist)  #    distortion parameter = (k_1,k_2,p_1,p_2,k_3)
        # print("rvecs:\n", rvecs)  # rotation matrix  # Extrinsic parameters
        # print("tvecs:\n", tvecs ) # translation vector  # Extrinsic parameters # useless for me in camera calibration

        self.IntrinsicMatrix = mtx
        self.distortionParameter = dist
        self.RotationMatrices = rvecs
        self.TranslationVectors = tvecs

        print("-----------------------------------------------------")

        # # Projection error (it reflects the quality of the calibrated photos) it is important, but I do not use it at all...
        tot_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error
        print(f"total error: {tot_error / len(obj_points)} ")
