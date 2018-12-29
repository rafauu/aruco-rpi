import numpy as np
import cv2
import cv2.aruco as aruco
import glob

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

cameraMatrix = np.array( [[1.01168235e+03,0.00000000e+00,6.23308512e+02],
                          [0.00000000e+00,1.00996685e+03,4.74372051e+02],
                          [0.00000000e+00,0.00000000e+00,1.00000000e+00]] )
distCoeffs = np.array( [1.33222613e-01,-3.51903491e-01,1.46665262e-04,8.83442353e-04, 3.03650225e-01] )
nx = 6
ny = 6
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
objpoints = []
imgpoints = []
firstMarkerId = 30

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.GridBoard_create(5, 7, 0.04, 0.01, aruco_dict)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    gray = aruco.drawDetectedMarkers(gray, corners, ids)

    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs)
    print("retval: ",retval)

    if retval != 0:
        gray = aruco.drawAxis(gray, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
        if firstMarkerId in ids:
            rmat = cv2.Rodrigues(rvec)[0]
            P = np.hstack((rmat,tvec))
            angles = -cv2.decomposeProjectionMatrix(P)[6]
            print("pitch: ", angles[0][0])
            print("roll:  ", angles[1][0])
            print("yaw:   ", angles[2][0])
            print("x:   ", tvec[0][0])
            print("y:   ", tvec[1][0])
            print("z:   ", tvec[2][0])

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
