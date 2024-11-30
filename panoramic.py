import cv2
import numpy as np

# Intrinsic matrices and distortion coefficients
left_camera_matrix = np.array([[417.066168961892, 0, 316.308968637001], 
                               [0, 413.806570743947, 183.531933911402], 
                               [0, 0, 1]])  

central_camera_matrix = np.array([[346.073691605571, 0, 325.071993184885], 
                                  [0, 350.210002456771, 186.679892053211], 
                                  [0, 0, 1]])  

right_camera_matrix = np.array([[376.432161960016, 0, 332.485458194884], 
                                [0, 379.653350082466, 180.778689124949], 
                                [0, 0, 1]])  

left_dist_coeffs = np.array([-0.422939296311705, 0.160822277856802, 0, 0, 0]) 
central_dist_coeffs = np.array([-0.292279580201782, 0.0699259295356006, 0, 0, 0]) 
right_dist_coeffs = np.array([-0.327106762491322, 0.0917546952095338, 0, 0, 0]) 

left_input_video = "rtsps://192.168.1.1:7441/p92hF5pL184MoRqI?enableSrtp"
central_input_video = "rtsps://192.168.1.1:7441/R1sQSttkap6L8Ypn?enableSrtp"
right_input_video = "rtsps://192.168.1.1:7441/qn0SqiLM0NR1OADS?enableSrtp"

cap_left = cv2.VideoCapture(left_input_video)
cap_central = cv2.VideoCapture(central_input_video)
cap_right = cv2.VideoCapture(right_input_video)

# Define homographies 
H_left_to_center = np.array([
    [ 1.01980091e+00,  7.31958059e-03, -2.62779709e+00],
    [-7.89500927e-03,  9.42679590e-01, -2.08025126e+02 +250],
    [ 1.63501591e-04, -1.00068629e-04,  1.00000000e+00]
    
])

# H_right_to_center = np.array([
#     [ 4.15241115e-01, -2.48649566e+00, 2.36765639e+02],
#     [-1.56746520e-01, -3.38485352e-01, 2.76727052e+02],
#     [-5.22221094e-04, -2.56142215e-03, 1.00000000e+00]
# ])


# New camera matrix for undistortion
new_left_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(left_camera_matrix, left_dist_coeffs, (640, 360), 1, (640, 360))
new_right_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(right_camera_matrix, right_dist_coeffs, (640, 360), 1, (640, 360))
new_central_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(central_camera_matrix, central_dist_coeffs, (640, 360), 1, (640, 360))

while True:
    # Reading frames from the cameras
    ret_left, frame_left = cap_left.read()
    ret_central, frame_central = cap_central.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_central or not ret_right:
        print("Error: Cannot read video stream")
        break
    
    # Undistort the frames
    undistorted_left = cv2.undistort(frame_left, left_camera_matrix, left_dist_coeffs, None, new_left_camera_matrix)
    undistorted_central = cv2.undistort(frame_central, central_camera_matrix, central_dist_coeffs, None, new_central_camera_matrix)
    #undistorted_right = cv2.undistort(frame_right, right_camera_matrix, right_dist_coeffs, None, new_right_camera_matrix)

    # # Apply homographies to align the frames
    left_warped = cv2.warpPerspective(undistorted_left, H_left_to_center, (640, 1080))
    # right_warped = cv2.warpPerspective(undistorted_right, H_right_to_center, (640, 360))
    # cv2.imshow('left_warped', left_warped)
    
    
    #rotate cameras
    left_2 = cv2.flip(cv2.rotate(undistorted_left, cv2.ROTATE_90_CLOCKWISE),1)
    left = cv2.flip(cv2.rotate(left_warped, cv2.ROTATE_90_CLOCKWISE),1)
    central = cv2.flip(cv2.rotate(undistorted_central, cv2.ROTATE_90_CLOCKWISE), 1)
    
    left_copy = left.copy()
    
    cv2.imshow('left_2', left)
    
    start_x = 250
    end_x = start_x + central.shape[1]
    end_y = central.shape[0]
    
    roi = left[0:end_y, start_x:end_x]

    alpha = 0.0
    beta = 1
    blended_region = cv2.addWeighted(roi, alpha, central, beta, 0)
    left[0:end_y, start_x:end_x] = blended_region
    
    mask_black = np.all(left == [0, 0, 0], axis=-1)
    left[mask_black] = left_copy[mask_black]
    
    #cv2.imshow('left', left_copy)
    cv2.imshow("Imagen combinada", left)
  

    #cv2.imshow('Panorama', panorama)
    cv2.imshow('central', central)

    
    

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_left.release()
cap_central.release()
cap_right.release()
#out.release()
cv2.destroyAllWindows()