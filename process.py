#!/usr/bin/env python

import cv2
import sys
import os
import numpy as np

# play nice with others
os.nice(5)

#filename = './input/Slooh_T3__2011-10-30T004517UTC.avi'
filename = './input/Quarter Moon Two AVI-Qu5XIpCdGLY.mp4'
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
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    rows,cols,depth = img1.shape

    return M
    



# get the first img
retval, dest_img = cap.read()

if not retval:
    print 'could not open video'
    sys.exit(-1)
    
rows,cols,depth = dest_img.shape
num_frames = 1
frame_no = 0
img_stats = {}

maxframes = np.nan
percentkeep = 0.01
# for every frame left
while retval:
    retval,image = cap.read()
    if retval:
        frame_no = frame_no+1
        if frame_no > maxframes:
            break;
        M = get_transformed_img(image,dest_img)
        
        if M is None:
            continue

        warped_img = cv2.warpPerspective(image, M, (cols,rows))
        lap_img = cv2.Laplacian(warped_img,cv2.CV_64F)
        mean_lap = np.mean(np.abs(lap_img))
        max_lap = np.max(np.abs(lap_img))
        img_stats[frame_no] = (M,max_lap,mean_lap)
        

        num_frames += 1
        print 'processed frame %d' % num_frames


cap.release()

max_lap = max(lap for M,lap,_ in img_stats.values())

good_frames = sorted(img_stats.items(), key=lambda item: item[1][2])
total_frames = len(good_frames)
num_selected_frames = int(np.ceil(percentkeep*total_frames))
print 'selecting {0} frames'.format(num_selected_frames)

good_frames = [val[0] for val in good_frames[-num_selected_frames:]]
assert len(good_frames) == num_selected_frames

# ok, now that we got max laplacian, we can reprocess with sharpness

cap = cv2.VideoCapture(filename)
retval = cap.open(filename)

# throw away first picture
cap.read()
summed_img = np.zeros(dest_img.shape)
sum_weights = np.zeros(dest_img.shape)
retval = True
frame_no = 0
while retval:
    retval,image = cap.read()
    if retval:
        frame_no+=1
        
        if frame_no > maxframes:
            break;

        if frame_no not in good_frames:
            continue
        M,lap,_ = img_stats[frame_no]
        warped_img = cv2.warpPerspective(image, M, (cols,rows))
        lap_img = np.abs(cv2.Laplacian(warped_img,cv2.CV_64F))
        lap_img = lap_img / max_lap
        #lap_img = np.ones(lap_img.shape)

        summed_img += np.float64(warped_img)*lap_img*256
        sum_weights += lap_img

        print 'added frame %d' % frame_no


sum_weights[sum_weights==0] = 1

out_img = summed_img/sum_weights

out_img = np.uint16(np.round(out_img))

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)
cv2.imwrite('out.tif',out_img)
