import cv2
import numpy as np

#Intrinsic matrixes

#Left camera
left_camera_matrix = np.array([[417.066168961892, 0,                316.308968637001], 
                               [0,               413.806570743947,  183.531933911402], 
                               [0,                0,                1]])  

#central camera
central_camera_matrix = np.array([[346.073691605571, 0,                325.071993184885], 
                               [0,               350.210002456771,  186.679892053211], 
                               [0,                0,                1]])  

#right camera
right_camera_matrix = np.array([[376.432161960016, 0,                332.485458194884], 
                               [0,               379.653350082466,  180.778689124949], 
                               [0,                0,                1]])  


#disttortion coefficients
# distortion coeffients = [k1, k2. p1, p2, k3]
left_dist_coeffs = np.array([-0.422939296311705, 0.160822277856802, 0, 0, 0]) 
central_dist_coeffs = np.array([-0.292279580201782, 0.0699259295356006, 0, 0, 0]) 
right_dist_coeffs = np.array([-0.327106762491322, 0.0917546952095338, 0, 0, 0]) 


left_input_video = "rtsps://192.168.1.1:7441/p92hF5pL184MoRqI?enableSrtp"
central_input_video = "rtsps://192.168.1.1:7441/R1sQSttkap6L8Ypn?enableSrtp"
right_input_video = "rtsps://192.168.1.1:7441/qn0SqiLM0NR1OADS?enableSrtp"

cap_left = cv2.VideoCapture(left_input_video)
cap_central = cv2.VideoCapture(central_input_video)
cap_right = cv2.VideoCapture(right_input_video)

#new camera matrix for undistortion
new_left_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(left_camera_matrix, left_dist_coeffs, (640, 360), 1, (640, 360))
new_right_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(right_camera_matrix, right_dist_coeffs, (640, 360), 1, (640, 360))
new_central_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(central_camera_matrix, central_dist_coeffs, (640, 360), 1, (640, 360))

#save video
output_video = "left.avi"

# Get the width, height, and FPS 
frame_width = 640
frame_height = 320
fps = cap_left.get(cv2.CAP_PROP_FPS)

# defining codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

while True:
    #reading cameras
    ret_left, frame_left = cap_left.read()
    ret_central, frame_central = cap_central.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_central or not ret_right:
        print("Error: Cannot read video stream")
        break
    
    #undistortion
    undistorted_left_frame = cv2.undistort(frame_left, left_camera_matrix, left_dist_coeffs, None, new_left_camera_matrix)
    undistorted_central_frame = cv2.undistort(frame_central, central_camera_matrix, central_dist_coeffs, None, new_central_camera_matrix)
    undistorted_right_frame = cv2.undistort(frame_right, right_camera_matrix, right_dist_coeffs, None, new_right_camera_matrix)
    
    #rotate cameras
    left = cv2.flip(cv2.rotate(undistorted_left_frame, cv2.ROTATE_90_CLOCKWISE),1)
    central = cv2.flip(cv2.rotate(undistorted_central_frame, cv2.ROTATE_90_CLOCKWISE), 1)
    right = cv2.flip(cv2.rotate(undistorted_right_frame, cv2.ROTATE_90_CLOCKWISE), 1)
    
    out.write(left)
    
    # combine cameras
    top_row = np.hstack((left, central, right))
    cv2.imshow("CameraVideo", top_row)
    
    cv2.imshow("Left", undistorted_left_frame)
    cv2.imshow("Right", undistorted_right_frame)
    cv2.imshow("Central", undistorted_central_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_central.release()
cap_right.release()
out.release()
cv2.destroyAllWindows()
    
    