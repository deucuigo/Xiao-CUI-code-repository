"""
动态摄像驱动
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_process import get_laser


class Capture:

    def __init__(self, fliename_camera, fliename_system):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.plane = None
        self.rmat = None
        self.tvecs = None

        # # load the camera parameter
        parameter_Intrinsic = np.load(fliename_camera)
        self.camera_matrix = parameter_Intrinsic['IntrinsicMatrix']
        self.dist_coeffs = parameter_Intrinsic['distortionParameter']
        print("\n##############################################")
        print("##Camera parameters is loaded for the file.##")
        print("##############################################\n")
        print("intrinsic matrix of camera:\n", self.camera_matrix)  # intrinsic parameters
        print("distortion parameter:\n", self.dist_coeffs)  # distortion parameter

        # # load the system parameter
        parameter_System = np.load(fliename_system)
        self.plane = parameter_System['plane']
        print("\n##############################################")
        print("##System parameters is loaded for the file.##")
        print("##############################################\n")
        print("the parameter of laser plane(a,b,c):\n", self.plane.reshape(1, -1)[0])  # intrinsic parameters

    def capture(self, out = False):  # ,rmx, tvec
        cap = cv2.VideoCapture(0)  # select camera, normally 0.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # set the image size of capturing
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        mtx = self.camera_matrix
        dist = self.dist_coeffs
        plane = self.plane


        # start camera
        success, img = cap.read()
        plt.ion()
        while True:
            success, img = cap.read()
            # print(img.shape)  # check the shape of the image

            img = cv2.undistort(img, mtx, dist)
            scale = 0.5
            newSize = tuple([int(img.shape[1] * scale), int(img.shape[0] * scale)])

            imgnew = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("live!", imgnew)  # output the video

            # # extracting the laser from image
            laser_cor = get_laser(img, method='skeleton')
            # print(type(laser_cor))
            u = [i[0] for i in laser_cor]
            v = [-i[1] for i in laser_cor]

            # # transform the coordinate to the reference board
            zc = self.transform_to_worldcor(u, v, mtx, plane, self.rmat, self.tvecs)

            if out == True:
                print(zc)

            # show it as plot on the scream
            # plt.cla()
            # plt.xlim([0, img.shape[1]])
            # plt.plot(range(0,len(zc)), -zc, '.')
            # plt.ylim(0,25)
            # plt.xlim(0, 500)
            # plt.show()

            key = cv2.waitKey(5)

            if key == 27:
                break

    def transform_to_worldcor(self, u, v, mtx, plane, rmx, tvec):
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        u0 = mtx[0, 2]
        v0 = mtx[1, 2]

        a = plane[0]
        b = plane[1]
        c = plane[2]

        Xc = (c * fy * (u - u0)) / (fx * fy - a * fy * u + a * fy * u0 - b * fx * v + b * fx * v0)
        Yc = (c * fx * (v - v0)) / (fx * fy - a * fy * u + a * fy * u0 - b * fx * v + b * fx * v0)
        Zc = (c * fx * fy) / (fx * fy - a * fy * u + a * fy * u0 - b * fx * v + b * fx * v0)

        # return Zc
        Zw = []
        for (ix, iy, iz) in zip(Xc, Yc, Zc):
            Ac = np.array([ix, iy, iz]).reshape(-1, 1)
            Aw = np.linalg.solve(rmx, Ac - tvec)
            z = Aw[-1]
            Zw.append(z)
        # return Zw
        return np.array(Zw)


    def define_zero_plane(self, path_img, cbrow=5, cbcol=7, cbsize=15, accuracy=0.001):
        img = cv2.imread(path_img)

        camera_matrix = self.camera_matrix
        dist_coeffs = self.dist_coeffs

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cbsize, accuracy)
        ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # draw the corner point on the chessboard
            img_mtx = cv2.drawChessboardCorners(gray, (cbrow, cbcol), corners2, ret)
            cv2.imshow("get matrices", img_mtx)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        model_points = np.zeros((cbcol * cbrow, 3), np.float32)
        model_points[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)
        model_points = model_points * cbsize
        # print(model_points)
        image_points = corners2
        retval, rvecs, tvecs = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)  # 使用直接照片处理，不需要提前矫正
        rmat = cv2.Rodrigues(rvecs)[0]

        self.rmat = rmat
        self.tvecs = tvecs

        return rmat, tvecs


