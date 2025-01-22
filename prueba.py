#!/usr/bin/env python -c

# Import OpenCV and threading packages
import cv2
import threading

# Define class for the camera thread.
class CamThread(threading.Thread):

    def __init__(self, previewname, camid):
        threading.Thread.__init__(self)
        self.previewname = previewname
        self.camid = camid

    def run(self):
        print("Starting " + self.previewname)
        previewcam(self.previewname, self.camid)

# Function to preview the camera.
def previewcam(previewname, camid):
    cv2.namedWindow(previewname)
    cam = cv2.VideoCapture(camid)

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('right_output.avi', fourcc, 20.0, (frame_width, frame_height))

    print(frame_height)
    print(frame_width)

    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewname, frame)
        rval, frame = cam.read()

        # write the flipped frame
        out.write(frame)

        key = cv2.waitKey(20)
        if key == 27:  
            break

    #release resources 
    cam.release()
    out.release()
    cv2.destroyWindow(previewname)

#thread1 = CamThread("Left camera", "rtsps://192.168.1.1:7441/p92hF5pL184MoRqI?enableSrtp")
thread1 = CamThread("Central camera", "rtsps://192.168.1.1:7441/SvXhKj5l4PfiSp5g?enableSrtp")
#thread1 = CamThread("Right camera", "rtsps://192.168.1.1:7441/qn0SqiLM0NR1OADS?enableSrtp")
thread1.start()

