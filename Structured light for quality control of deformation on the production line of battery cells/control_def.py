import numpy as np
from cali_cameras import Cali_Camera
from cali_lasers import Cali_lasers
from capturings import Capture
from image_process import get_laser, get_Picture

path_folder = r"D:\python\bild\images\calibration\hand on"

type_image_cam_cali = "png"
cam_file = "parameter_Intrinsic.npz"
sys_file = "parameter_system.npz"

type_image_sys_cali = type_image_cam_cali
path_cam_cali = f"{path_folder}\cali_cameras"
path_sys_cali = f"{path_folder}\cali_systems"



re_camera = input("Do you want to calibrate Camera? (Y/N)")
if re_camera.lower() == 'y':
    camera = Cali_Camera()
    print("###############################################################")
    re_newPict = input("Do you want to take new images to calibrate the cameras? (Y/N)")
    if re_newPict.lower() == 'y':
        print("space -- taking picture; esc -- finish taking pictures")
        get_Picture("camera", folder_path=path_cam_cali)
        print("###############################################################")
        camera.cali_camera_python(pict_add=f"{path_cam_cali}\*.{type_image_cam_cali}", cbrow_in=5,
                                  cbcol_in=9, cbsize_in=25, accuracy_in=0.001)
        camera.save_parameter(cam_file)
    else:
        camera.cali_camera_python(pict_add=f"{path_cam_cali}_\*.{type_image_cam_cali}", cbrow_in=5,
                                  cbcol_in=9, cbsize_in=25, accuracy_in=0.001)
        camera.save_parameter(cam_file)

print("\n\n###############################################################")
re_system = input("Do you want to calibrate the system? (Y/N)")
if re_system.lower() == 'y':
    sys = Cali_lasers(fliename_camera="parameter_Intrinsic.npz", solving_method="cross-ratio")
    re_newPict = input("Do you want to take new images to calibrate the cameras? (Y/N)")
    if re_newPict.lower() == 'y':
        print("space -- taking picture; esc -- finish taking pictures")
        get_Picture("system", folder_path=path_sys_cali)
        print("###############################################################")
        sys.cali_system(path_folder=path_sys_cali, type_image= type_image_sys_cali)
        sys.save_parameter(sys_file)
    else:
        sys.cali_system(path_folder=f"{path_sys_cali}_", type_image= type_image_sys_cali)
        sys.save_parameter(sys_file)

print("\n\n###############################################################")

print("###############################################################")
print("Start measurement")

measurement = Capture(fliename_camera="parameter_Intrinsic.npz", fliename_system="parameter_system.npz")

print("\n\n###############################################################")
print("set reference plane")
re_plane = input("Do you want to take a new reference plane? (Y/N)")
print("check whether the chessboard is correctly processed, and press the Space")
if re_plane.lower() == 'y':
    get_Picture("plane", folder_path= f"{path_folder}\cali_plane")
measurement.define_zero_plane(path_img=f"{path_folder}\cali_plane\cali0.png", cbrow=5,
                              cbcol=7, cbsize=15, accuracy=0.001)
print("\n\n###############################################################")
measurement.capture(out=False)
