from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
import matplotlib.pyplot as plt
# import serial

# ser = serial.Serial('/dev/cu.usbmodem1411', 115200)
# /dev/cu.usbmodem1411

#python 35

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
# ap.add_argument("-p", "--port", help="port for the serial connection")
args = vars(ap.parse_args())

# for orange pingpong ball
# 0,0,143 255,255,255
# 0,111,209 255,206,255
lowerBound = (0, 111, 209)
lowerBound = (0, 182, 162)
# 0,182,162 255,255,255
upperBound = (255, 206, 255) ## set upper lower bound for color threshold 
upperBound = (255, 255, 255) ## set upper lower bound for color threshold 
pts = deque(maxlen=args["buffer"])

def testFPS():
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

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    # camera.set(cv2.CAP_PROP_FPS, 15)

    # if int(major_ve)  < 3 :
    #     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #     print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    # else :
    
    # testFPS()
else:
    camera = cv2.VideoCapture(args["video"])



i = 0 
j = 0
start = time.time()
num_frames_2 = 1200
x_arr = []
y_arr = []


# while True:
while True:

    (grabbed, frame) = camera.read()
    j += 1
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

    frame = imutils.resize(frame, width=800)
    # frame = cv2.resize(frame, (640, 480))
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lowerBound, upperBound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
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

    for i in range(1, len(pts)):
    # if either of the tracked points are None, ignore
    # them
        if pts[i - 1] is None or pts[i] is None:    
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    
    if center is None:
        str_x, str_y = "x---", "y---"
    else:
        str_x, str_y = "x{:03d}".format(center[0]), "y{:03d}".format(center[1])

    
    # print(j, str_x, str_y)
    x_arr.append(int(str_x[1:]))
    y_arr.append(int(str_y[1:]))
    # ser.write(str_x.encode())
    # ser.write(str_y.encode())
    

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

end = time.time()
seconds = end - start
print("Simulated fps:", i / seconds)

x_arr_np = np.array(x_arr)
x_arr_np = x_arr_np /  (i / seconds)
y_arr_np = np.array(y_arr)
y_arr_np = y_arr_np /  (i / seconds)

fps = int(i / seconds)

plt.figure(1)


plt.subplot(211)
plt.title('Vertical Position')
# plt.xlabel('Frame Number @ 30FPS')
plt.ylabel('Position')
# plt.axhline(y=300, linewidth=1, color='r', ls='--', label="error")
plt.plot(range((len(x_arr))), x_arr,label="actual")

fs = 100 # sample rate 
f = 0.8 # the frequency of the signal
x = np.arange(len(x_arr))

y = [ 40*np.sin(2*np.pi*f * (i/fs) + 2) + 273 - x_arr[i] for i in x]
# plt.plot(x,y, linewidth=1, color='r', ls='--', label="reference")
plt.legend()





plt.subplot(212)
plt.title('Horizontal Position')


plt.xlabel('Frame Number @ 30FPS') 
plt.ylabel('Position')
# plt.axhline(y=238, linewidth=1, color='r', ls='--', label="reference")


fs = 100
 # sample rate 
f = 0.78 # the frequency of the signal
x = np.arange(len(x_arr))
y = [ 40*np.sin(2*np.pi*f * (i/fs) + np.pi*0.32) + 220 - y_arr[i] for i in x]
plt.plot(range(len(x_arr)), y_arr, label="actual")
# plt.plot(x,y, linewidth=1, color='r', ls='--', label="error")


plt.legend()
# plt.yticks(np.linspace(min(x), max(x)+1, 5.0))

plt.show()

camera.release()
cv2.destroyAllWindows()

