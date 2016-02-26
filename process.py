#!/usr/bin/env python

import cv2
import sys
import os
import numpy as np


def get_transformed_img(img1,img2):
    """returns img1 aligned to img2"""
    si1 = np.copy(img1)
    si1 = np.uint8(si1)
    si2 = np.copy(img2)
    si2 = np.uint8(si2)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)

    kp1, des1 = sift.detectAndCompute(si1,None)
    kp2, des2 = sift.detectAndCompute(si2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good) < 10:
        return None

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    
    rows,cols,depth = img1.shape

    return M
    

# play nice with others
os.nice(5)

#filename = './input/Slooh_T3__2011-10-30T004517UTC.avi'
#filename = './input/First Moon AVI-dxNcRnrnCSA.mp4'
#filename = "./input/moon_short.MOV"
#filename = "./input/andy_moon_l.MOV"
filename = "./input/moon_test.mp4"

cap = cv2.VideoCapture(filename)
retval = cap.open(filename)

print 'Calculating best frame...'
laplacians = []
max_lap = -1
best_frame = -1
best_img = None
cur_frame = 0
while retval:
    retval, img = cap.read()
    if not retval:
        continue
    lap_img = cv2.Laplacian(img,cv2.CV_32F)
    mean_lap = np.mean(np.abs(lap_img))
    laplacians.append(mean_lap)

    if mean_lap > max_lap:
        best_frame = cur_frame
        best_img = img

    cur_frame += 1
    
# get the first img

assert best_img is not None

print 'Best frame is {0}'.format(best_frame)
cap.release()

percentkeep = 0.1
sorted_laplacians = sorted(enumerate(laplacians), key=lambda (i,lap): lap)
num_selected_frames = int(np.ceil(percentkeep*cur_frame))
good_frames = set([val[0] for val in sorted_laplacians[-num_selected_frames:]])

print 'selecting {0} frames'.format(num_selected_frames)
print good_frames
assert len(good_frames) == num_selected_frames

rows,cols,depth = best_img.shape
num_frames = 1
frame_no = 0
img_stats = {}

maxframes = np.nan


# ok, now that we got max laplacian, we can reprocess with sharpness
cap = cv2.VideoCapture(filename)
retval = cap.open(filename)

summed_img = np.zeros(best_img.shape)
sum_weights = np.zeros(best_img.shape)
retval = True
frame_no = 0
while retval:
    retval,image = cap.read()
    if retval:
        
        if frame_no > maxframes:
            break;

        if frame_no not in good_frames:
            frame_no+=1
            continue
        M = get_transformed_img(image,best_img)
        warped_img = cv2.warpPerspective(image, M, (cols,rows))
        lap_img = np.abs(cv2.Laplacian(warped_img,cv2.CV_64F))
        lap_img = lap_img / max_lap
        #lap_img = np.ones(lap_img.shape)

        summed_img += np.float64(warped_img)*lap_img*256
        sum_weights += lap_img

        print 'added frame %d' % frame_no
        frame_no+=1

sum_weights[sum_weights==0] = 1

out_img = summed_img/sum_weights

out_img = np.uint16(np.round(out_img))

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)
cv2.imwrite('out.tif',out_img)
