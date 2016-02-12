#!/usr/bin/env python

import cv2
import sys
import os
import numpy as np

# play nice with others
os.nice(5)

#filename = './input/Slooh_T3__2011-10-30T004517UTC.avi'
filename = './input/First Moon AVI-dxNcRnrnCSA.mp4'
#filename = "./input/moon1.avi"
#filename = "./input/moon2.avi"
cap = cv2.VideoCapture(filename)
retval = cap.open(filename)
retval = True


def get_transformed_img(img1,img2):
    """returns img1 aligned to img2"""
    si1 = np.copy(img1)
    si1 = np.uint8(si1)
    si2 = np.copy(img2)
    si2 = np.uint8(si2)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)

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
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    rows,cols,depth = img1.shape

    return cv2.warpPerspective(img1, M, (cols,rows))


# get the first img
retval, dest_img = cap.read()

if not retval:
    print 'could not open video'
    sys.exit(-1)
    
summed_img = np.float32(dest_img)

num_frames = 1

# for every frame left
while retval:
    retval,image = cap.read()
    if retval:
        print 'processed frame %d' % num_frames
        transformed_img = get_transformed_img(image,dest_img) 
        if transformed_img is None:
            continue

        summed_img += np.float32(transformed_img)
        num_frames += 1
        
out_img = np.uint8(np.round(summed_img/num_frames))
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)
cv2.imwrite('out.tif',out_img)
