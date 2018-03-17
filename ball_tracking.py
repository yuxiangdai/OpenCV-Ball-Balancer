from collections import deque
import numpy as np
import argparse
# import imutils
import cv2
import time
# import serial

# ser = serial.Serial('/dev/tty.usbserial', 9600)


#python 35

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

yellowLB = (20, 100, 100)
yellowUB = (30, 255, 255) ## set upper lower bound for color threshold 
# greenLower = (51, 60, 60)
# greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    # camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture(1)
    # camera.set(cv2.CAP_PROP_FPS, 15)

    # camera.set(cv2.CAP_PROP_FPS, 10)







    #############
    # Test Code


    # if int(major_ve)  < 3 :
    #     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #     print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    # else :
    test = camera.get(cv2.CAP_PROP_FPS)
    print("Frames per second should be : {0}".format(test))

    # Number of frames to capture
    num_frames = 120;
    print("Capturing {0} frames".format(num_frames))

    # Start time
    start = time.time()
        
    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = camera.read()
        
    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps))
else:
    camera = cv2.VideoCapture(args["video"])



i = 0 
j = 0
start = time.time()
num_frames_2 = 120

# while True:
for i in range(0, num_frames_2):

    # j += 1
    # if(j % 120 == 0):
    #     j = 0
    #     end = time.time()
    #     seconds = end - start
    #     print("Simulated fps:", 120 / seconds)
    #     start = end  ## this ignores the time it takes to print statement
      
    (grabbed, frame) = camera.read()
    # # this is really dumb fix it
    # if(i == 1):
    #     i = 0
    # else:
    #     i = 1

    # if(i == 1):
    #     j += 1
    #     (grabbed, frame) = camera.read()  ## right now just grabs the same frame twice


    # if(j % 120 == 0):
    #     j = 0
    #     end = time.time()
    #     seconds = end - start
    #     print("Simulated fps:", 60 / seconds)
    #     start = end  ## this ignores the time it takes to print statement
      

    if args.get("video") and not grabbed:
        break

    frame = cv2.resize(frame, (800, 600))

    # frame = imutils.resize(frame, width=800)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, yellowLB, yellowUB)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


    pts.appendleft(center)

    # print()

    for i in range(1, len(pts)):
    # if either of the tracked points are None, ignore
    # them
        if pts[i - 1] is None or pts[i] is None:    
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(1) & 0xFF


    print(center)
    if key == ord("q"):
        break

end = time.time()
seconds = end - start
print("Simulated fps:", num_frames_2 / seconds)


camera.release()
cv2.destroyAllWindows()

