import cv2
import numpy as np


# #Left Camera
# # calibration parameters
# fx = 417.066168961892
# fy = 413.806570743947
# cx = 316.308968637001
# cy = 183.531933911402

# # Distortion coeffients
# k1 = -0.422939296311705
# k2 = 0.160822277856802
# k3 = 0
# p1 = 0
# p2 = 0

#Central Camera
# calibration parameters
fx = 346.073691605571
fy = 350.210002456771
cx = 325.071993184885
cy = 186.679892053211

# Distortion coeffients
k1 = -0.292279580201782
k2 = 0.0699259295356006
k3 = 0
p1 = 0
p2 = 0

# #Right Camera
# # calibration parameters
# fx = 376.432161960016
# fy = 379.653350082466
# cx = 332.485458194884
# cy = 180.778689124949

# # Distortion coeffients
# k1 = -0.327106762491322
# k2 = 0.0917546952095338
# k3 = 0
# p1 = 0
# p2 = 0


#camera matrix
camera_matrix = np.array([[fx, 0, cx], 
                         [0, fy, cy], 
                         [0, 0, 1]])  


#disttortion coefficients
dist_coeffs = np.array([k1, k2, p1, p2, k3])  


# Open the video file
#Left camera
#input_video = "rtsps://192.168.1.1:7441/p92hF5pL184MoRqI?enableSrtp"
input_video = "rtsps://192.168.1.1:7441/R1sQSttkap6L8Ypn?enableSrtp"
output_video = "central.avi"
output_video2 = "distorted_right_camera.avi"


cap = cv2.VideoCapture(input_video)


# Get the width, height, and FPS 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# defining codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))
out2 = cv2.VideoWriter(output_video2, fourcc, 20.0, (frame_width, frame_height))


# Create an optimal new camera matrix for undistortion
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (frame_width, frame_height), 1, (frame_width, frame_height))


delay = int(1000 / fps)


while cap.isOpened():
   ret, frame = cap.read()
   if not ret:
       break
   
   # Undistort the frame
   undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
   
   out.write(undistorted_frame)
   out2.write(frame)
   
   # Display the frame (optional)
   cv2.imshow('Undistorted Video', undistorted_frame)
   if cv2.waitKey(delay) & 0xFF == ord('q'):
       break


# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()