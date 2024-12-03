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
    [1.01980091, 7.31958059e-03, -2.62779709],
    [-7.89500927e-03, 9.42679590e-01, -208.025126 + 250],
    [1.63501591e-04, -1.00068629e-04, 1.0]
])

H_right_to_center = np.array([
    [1.02845693, 7.66146089e-03, -3.15966250],
    [2.81709251e-02, 1.08613596e+00, 231.720455 - 245],
    [2.94507790e-05, 1.90601076e-04, 1.0]
])

# New camera matrix for undistortion
new_left_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(left_camera_matrix, left_dist_coeffs, (640, 360), 1, (640, 360))
new_right_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(right_camera_matrix, right_dist_coeffs, (640, 360), 1, (640, 360))
new_central_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(central_camera_matrix, central_dist_coeffs, (640, 360), 1, (640, 360))

while True:
    # Reading frames from the cameras
    ret_left, frame_left = cap_left.read()
    ret_central, frame_central = cap_central.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_central or not ret_right:
        print("Error: Cannot read video stream")
        break
    
    # Apply undistortion to the frames
    undistorted_left = cv2.undistort(frame_left, left_camera_matrix, left_dist_coeffs, None, new_left_camera_matrix)
    undistorted_central = cv2.undistort(frame_central, central_camera_matrix, central_dist_coeffs, None, new_central_camera_matrix)
    undistorted_right = cv2.undistort(frame_right, right_camera_matrix, right_dist_coeffs, None, new_right_camera_matrix)

    # Apply homographies to align the frames
    left_warped = cv2.warpPerspective(undistorted_left, H_left_to_center, (640, 1080))
    right_warped = cv2.warpPerspective(undistorted_right, H_right_to_center, (640, 360))

    # Flip and rotate the frames only once per iteration
    # left = cv2.flip(cv2.rotate(left_warped, cv2.ROTATE_90_CLOCKWISE), 1)
    # right = cv2.flip(cv2.rotate(right_warped, cv2.ROTATE_90_CLOCKWISE), 1)
    # central = cv2.flip(cv2.rotate(undistorted_central, cv2.ROTATE_90_CLOCKWISE), 1)
    
    left_copy = left_warped.copy()
    
    # Combine sections into the final image
    left_warped[250:250 + undistorted_central.shape[0], 0:undistorted_central.shape[1]] = cv2.addWeighted(
        left_warped[250:250 + undistorted_central.shape[0], 0:undistorted_central.shape[1]], 0.0, undistorted_central, 1.0, 0)
    
    left_copy2 = left_warped.copy()

    left_warped[493:493 + undistorted_right.shape[0], 0:undistorted_right.shape[1]] = cv2.addWeighted(
        left_warped[493:493 + undistorted_right.shape[0], 0:undistorted_right.shape[1]], 0.0, undistorted_right, 1.0, 0)
    
    mask_black = np.all(left_warped == [0, 0, 0], axis=-1)
    left_warped[mask_black] = left_copy[mask_black]
    
    mask_black = np.all(left_warped == [0, 0, 0], axis=-1)
    left_warped[mask_black] = left_copy2[mask_black]
    
    cv2.imshow("Imagen combinada", left_warped)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_left.release()
cap_central.release()
cap_right.release()
cv2.destroyAllWindows()
