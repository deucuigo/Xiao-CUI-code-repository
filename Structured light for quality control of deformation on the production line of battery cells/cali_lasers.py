import cv2
import numpy as np
import glob

from image_process import get_laser




# 目前求出的结果还有误差，后面需要调整程序。
# 利用角点反求旋转位移矩阵需要使用原始图像
# 求解交点 角点和激光图像都需要修正 需要在矫正之后的图像上完成。

# 带着激光无法进行角点的读取，需要读入标定板图像，然后再读入激光图像，之后再进行激光标定。标定程序已经成熟了，现在需要做的就是确定其正在正确运行。
# 这时候需要做method处理一下
# 需要做成class使用，用来存储。这个文件就是method。 使用其他文件进行处理，将他们放在一起。

class Cali_lasers:

    def __init__(self, fliename_camera,solving_method):

        self.solving_method = solving_method
        self.camera_matrix = None
        self.dist_coeffs = None
        self.result = None
        self.cbrow = None
        self.cbcol = None
        self.cbsize = None
        self.accuracy = None
        self.result = None

        print("\n##########################################")
        print("##     Calibration of system start.     ##")
        print("##########################################\n")

        # # read in camera parameter
        if self.camera_matrix is None:
            parameter_Intrinsic = np.load(fliename_camera)
            self.camera_matrix = parameter_Intrinsic['IntrinsicMatrix']
            self.dist_coeffs = parameter_Intrinsic['distortionParameter']
            print("\n##############################################")
            print("##Camera parameters is loaded for the file.##")
            print("##############################################\n")
            print("intrinsic matrix of camera:\n", self.camera_matrix)  # intrinsic parameters
            print("distortion parameter:\n", self.dist_coeffs)  # distortion parameter

        # # determine the chessboard for system calibration
        print("\n\n##############################################")
        print("The default chessboard is showed as follow, ")
        print("  the number of corner points in rows: 5")
        print("  the number of corner points in columns: 7")
        print("  the size of grid: 12")
        print("  the extraction accuracy: 0.001")
        print("##############################################")
        default_chessboard = input("\nDo you want to use the default chessboard?(Y/N)")
        if default_chessboard.lower() == 'y':
            self.cbrow = 5
            self.cbcol = 7
            self.cbsize = 12
            self.accuracy = 0.001
        else:
            self.cbrow = input("New cbrow: ")
            self.cbcol = input("New cbcol: ")
            self.cbsize = input("New cbsize: ")
            self.accuracy = input("New accuracy: ")
        print("\n##############################################")
        print("## Chessboard parameter is showed as follow.##")
        print("##############################################\n")
        print("the number of corner points in rows:", self.cbrow)
        print("the number of corner points in columns:", self.cbcol)
        print("the size of grid:", self.cbsize)
        print("the extraction accuracy:", self.accuracy)
        print()

    def cali_system(self, path_folder, type_image):

        # ----------------------------------
        # # parameter input
        cbrow = self.cbrow
        cbcol = self.cbcol
        cbsize = self.cbsize
        accuracy = self.accuracy
        camera_matrix = self.camera_matrix
        dist_coeffs = self.dist_coeffs

        # ----------------------------------
        # # images imput
        path = f"{path_folder}\*.{type_image}"
        images = glob.glob(path)
        num_of_pict = len(images)

        print("#############################################################")
        print(f"the method of solving the 3D coordinate is {self.solving_method}")
        print("#############################################################\n")

        cor_camera_list = np.array([])
        for i in range(int(num_of_pict / 2)):
            print("\nthe processing picture is ", images[2 * i][-9:], 'and', images[2 * i + 1][-9:])
            print("please press space to calibrate next set of images")
            corCam = self.laserPointExtraction(camera_matrix, dist_coeffs, images[2 * i], images[2 * i + 1], cbrow,
                                               cbcol, cbsize, accuracy)
            print("the coordinate of the extracted laser point is ")
            print(corCam)
            # for i in range(len(corCam)-1):
            #     print(np.linalg.norm(corCam[i]-corCam[i+1]))

            if len(cor_camera_list) == 0:
                cor_camera_list = corCam[:]
            else:
                cor_camera_list = np.concatenate((cor_camera_list, corCam[:]), axis=0)
        # print(cor_camera_list.shape)
        print("\n# # the calibration of system has been done. # # \n")

        # # least square method ，get a,b,c and return them
        # cor_camera_list = np.array(cor_camera_list)

        """
        z = aX + by + c   [x,y,1]*n @ [a,b,c].T = [z]*n
        A = [x,y,1]
        x = cor_camera_list[:, 0]
        y = cor_camera_list[:, 1]
        """

        # print("!\n",cor_camera_list)

        z = cor_camera_list[:, 2].reshape(-1, 1)
        # print(z)
        # A = cor_camera_list[:,[0,1,3]]
        A = np.array([[i[0], i[1], 1] for i in cor_camera_list])
        # print(A)
        result = np.linalg.lstsq(A, z, rcond=None)

        # print(A)
        print(f"the result [a,b,c] is {result[0].reshape(1, -1)}")
        self.result = result[0]
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        # self.save_parameter()
        return  self.result


    def laserPointExtraction(self, camera_matrix, dist_coeffs, path_img1, path_img2, cbrow=5, cbcol=9, cbsize=20,
                             accuracy=0.001):
        img = cv2.imread(path_img1)
        img_laser = cv2.imread(path_img2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("distort",gray)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cbsize, accuracy)

        # -------------------------------------------------------------
        # get the rotation matrix and translation vector of the target
        # -------------------------------------------------------------

        # # get corner point from images
        ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # draw the corner point on the chessboard
            # img_mtx = cv2.drawChessboardCorners(gray, (cbrow, cbcol), corners2, ret)
            # print(corners2)
            # cv2.imshow("get matrices", img_mtx)

        # # using the PnP algorithm to get the the rotation matrix and translation vector
        model_points = np.zeros((cbcol * cbrow, 3), np.float32)
        model_points[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)
        model_points = model_points * cbsize
        # print(model_points)
        image_points = corners2
        retval, rvecs, tvecs = cv2.solvePnP(model_points, image_points, camera_matrix,
                                            dist_coeffs)  # directly using the image got from camera
        # print(corners2)
        # print(f"rvec = {rvecs}")
        # print(f"tvec = {tvecs*cbsize}")
        rmat = cv2.Rodrigues(rvecs)[0]
        # rtmatrix = np.vstack((np.hstack((rmat, tvecs * cbsize)), [0, 0, 0, 1]))
        rmat_inv = np.linalg.pinv(rmat)

        # # set the variable for later use
        # plane vector parameter - extract
        r31 = rmat_inv[2, 0]
        r32 = rmat_inv[2, 1]
        r33 = rmat_inv[2, 2]
        tz = (tvecs[-1])
        tx = (tvecs[0])
        ty = (tvecs[1])
        # print(tvecs)

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        u0 = camera_matrix[0, 2]
        v0 = camera_matrix[1, 2]

        # -------------------------------------------------------------
        # get the 2D coordinates of laser points
        # -------------------------------------------------------------

        # # get the corner point

        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        gray_undistorted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_undistorted, (cbrow, cbcol), None)
        if ret:
            corners2_new = cv2.cornerSubPix(gray_undistorted, corners, (11, 11), (-1, -1), criteria)
            # draw the corner point on the chessboard
            img = cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2_new, ret)
        # print(corners2_new)

        # # get the laser point

        img_laser = cv2.undistort(img_laser, camera_matrix, dist_coeffs)
        # cv2.imshow("laser", img_laser)
        fit_laser = get_laser(img_laser, method='skeleton')
        # # 标记激光线采样位置
        for i in fit_laser:
            cv2.circle(img_laser, (int(i[0]), int(i[1])), 1, (0, 0, 0))

        # # laser line fitting(laser line should be a straight line,so laser line can be fitted to decrease the noise)

        fit_laser_nom = np.polyfit(fit_laser[:, 0], fit_laser[:, 1], 1)  # fittinjg laser line
        fit_laser_nom_figure = np.polyfit(fit_laser[:, 1], fit_laser[:, 0],
                                          1)  # reverse fitting laser line # the position of x,y are changed

        # # draw the white fitting line for laser extraction
        y1 = 0
        x1 = int(np.polyval(fit_laser_nom_figure, y1))
        y2 = 1000
        x2 = int(np.polyval(fit_laser_nom_figure, y2))
        cv2.line(img_laser, (x1, y1), (x2, y2), (255, 255, 255), 1,
                 4)  # draw the white fitting line for laser extraction

        # # fitting the grid of chessboard

        fit_corners = corners2_new.reshape((cbcol, cbrow, 2))
        # print(fit_corners) # print all corner points
        cross_list = []
        for ir in range(cbrow):
            # # # print(fit_corners[:, ir, :])  # print all x of corner points
            fit_corners_nom = np.polyfit(fit_corners[:, ir, 0], fit_corners[:, ir, 1], 1)
            x1 = 100
            y1 = int(np.polyval(fit_corners_nom, 100))
            x2 = 1500
            y2 = int(np.polyval(fit_corners_nom, 1500))
            cv2.line(img_laser, (x1, y1), (x2, y2), (0, 255, 0), 2, 4)

            # # # get the cross points of laser line and grid line
            x_cross = -(fit_laser_nom[1] - fit_corners_nom[1]) / (fit_laser_nom[0] - fit_corners_nom[0])
            y_cross = np.polyval(fit_corners_nom, x_cross)
            # print(x_cross,y_cross) # show the cross point between chess board and laser

            # -------------------------------------------------------------
            # conversion of the 2D coordinates  to 3D coordinates of laser points
            # -------------------------------------------------------------
            if self.solving_method == "cross-ratio":
                # cross-ratio invariance method
                u = x_cross
                v = y_cross
                # print(u,v)
                a = fit_corners[0, ir, :]
                b = fit_corners[1, ir, :]
                d = fit_corners[-1, ir, :]
                ab = np.linalg.norm(a - b)
                ad = np.linalg.norm(a - d)
                ac = np.linalg.norm(a - [x_cross, y_cross])
                # print(a, b, d)
                # print(ab, ad, ac)
                cr = (ac / (ac - ab)) / (ad / (ad - ab))
                AD = cbsize * (cbcol - 1)
                AB = cbsize
                r = AD / (AD - AB)
                AC = (cr * r) / (cr * r - 1) * AB
                # print(AD,AB,AC)
                Yt = AC
                Xt = ir * cbsize
                # print(Xt,Yt,0)
                [Xc, Yc, Zc] = rmat @ [[Xt], [Yt], [0]] + tvecs

            if self.solving_method == "spatial geometry":
                # spatial geometry method
                Xc = (fy * (u - u0) * (r31 * tx + r32 * ty + r33 * tz)) / (
                            fx * fy * r33 + fy * r31 * u - fy * r31 * u0 + fx * r32 * v - fx * r32 * v0)
                Yc = (fx * (v - v0) * (r31 * tx + r32 * ty + r33 * tz)) / (
                            fx * fy * r33 + fy * r31 * u - fy * r31 * u0 + fx * r32 * v - fx * r32 * v0)
                Zc = (fx * fy * (r31 * tx + r32 * ty + r33 * tz)) / (
                            fx * fy * r33 + fy * r31 * u - fy * r31 * u0 + fx * r32 * v - fx * r32 * v0)

            cross_list.append([Xc[0], Yc[0], Zc[0]])

            # print(Xc, Yc, Zc)

            #  # the following method is only valid when camera is vertical to the board

            #     whole_length = np.linalg.norm(ifit[0] - ifit[-1])
            #     cross_length = np.linalg.norm(ifit[0] - [x_cross, y_cross])
            #     x_cross_target = cross_length / whole_length * (cbrow - 1) * cbsize
            #     y_cross_target = nfit * cbsize
            #     cross_list.append([x_cross_target, y_cross_target])

            # whole_length = np.linalg.norm(fit_corners[0, ir] - fit_corners[-1, ir])
            # cross_length = np.linalg.norm(fit_corners[0, ir] - [x_cross, y_cross])
            # x_cross_target = ir * cbsize
            # y_cross_target = cross_length / whole_length * (cbcol - 1) * cbsize
            # print(cross_length,whole_length)
            # print(x_cross_target,y_cross_target)

            # # # draw the  intersection on the scream

            cv2.circle(img_laser, (int(x_cross), int(y_cross)), 5, (255, 255, 255), 2)

        cv2.imshow("img", img_laser)
        cv2.waitKey(0)
        return np.array(cross_list)


    def save_parameter(self,path_system= "parameter_system.npz"):
        np.savez("parameter_system.npz",plane = self.result )
        print("\n################################")
        print("##Parmeter of system is saved.##")
        print("################################")
