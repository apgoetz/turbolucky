#!/usr/bin/env python

import cv2
import sys
import os
import numpy as np
import drizzle
import _drizzle

is16=False

detector = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
#detector = cv2.ORB_create()

def get_transformed_img(img1,img2_kp, img2_des):
    """returns img1 aligned to img2"""
    
    si1 = np.copy(img1)
    si1 = np.uint8(si1)


    kp1, des1 = detector.detectAndCompute(si1,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,img2_des, k=2)

    # Apply ratio test
    good = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good) < 10:
        print 'could not find good alignment: {0}'.format(len(good))
        return None

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ img2_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    
    rows,cols,depth = img1.shape

    return np.float32(M)
    

# play nice with others
os.nice(5)

#filename = './input/Slooh_T3__2011-10-30T004517UTC.avi'
#filename = './input/First Moon AVI-dxNcRnrnCSA.mp4'
#filename = "./input/moon_short.MOV"
#filename = "./input/andy_moon_l.MOV"
#filename = "./input/moon_test.mp4"
filename = "./input/low_res.mov"
outname = 'out.tif'
percentkeep = 0.01
pixfrac = 0.5
scalefrac = 0.4

if len(sys.argv) > 1:
    filename = sys.argv[1]

if len(sys.argv) > 2:
    outname = sys.argv[2]

if len(sys.argv) > 3:
    percentkeep = float(sys.argv[3])
    assert percentkeep > 0 and percentkeep <= 1

if len(sys.argv) > 4:
    pixfrac = float(sys.argv[4])
    assert pixfrac > 0 and pixfrac <= 1

if len(sys.argv) > 5:
    scalefrac = float(sys.argv[5])
    assert scalefrac > 0 and scalefrac <= 1



print "input:\t{0}\noutput:\t{1}\nlucky percentage:\t{2}\npixfrac:\t{3}\nscalefrac:\t{4}\n".format(filename,outname,percentkeep, pixfrac,scalefrac)
    
cap = cv2.VideoCapture(filename)
retval = cap.open(filename)

print 'Calculating best frame...'
laplacians = []
tot_max_lap = -1
max_lap = -1
best_frame = -1
best_img = None
cur_frame = 0
while retval:
    retval, img = cap.read()
    if not retval:
        continue
    lap_img = cv2.Laplacian(img,cv2.CV_32F)
    tot_lap = np.sum(np.abs(lap_img))

    laplacians.append(tot_lap)

    if tot_lap > tot_max_lap:
        best_frame = cur_frame
        best_img = np.copy(img)
        max_lap = np.max(np.abs(lap_img))
        tot_max_lap = tot_lap

    cur_frame += 1
    
# get the first img

assert best_img is not None


print 'Best frame is {0}'.format(best_frame)

best_kp,best_des = detector.detectAndCompute(np.uint8(best_img),None)

# extend image in either direction
percent_increase = 0.25 # how much to increase the picture
rows,cols,depth = best_img.shape
p_rows = int(rows*percent_increase/2)
p_cols = int(cols*percent_increase/2)

new_img = np.zeros((rows + 2*p_rows, cols+2*p_cols,depth))
new_img[p_rows:p_rows+rows,p_cols:p_cols+cols] = best_img
best_img = new_img

cap.release()


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

# parameters of drizzle algo

drizzled_img = drizzle.build_dest_array(best_img, pixfrac, scalefrac)
weight_img = np.zeros(drizzled_img.shape, dtype=np.float32)


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
        M = get_transformed_img(image,best_kp, best_des)
        if M is None:
            print 'skipping bad frame {0}'.format(frame_no)
            frame_no+=1
            continue
        lap_img = np.abs(cv2.Laplacian(image,cv2.CV_32F))
        max_lap = np.max(lap_img)
        lap_img = np.float32(lap_img / max_lap)

        # do that drizzle
        image = np.float32(image)
        if not is16:
            image *= 256
            
        _drizzle.drizzle(image,drizzled_img,M,weight_img,lap_img,pixfrac,scalefrac)
        
        print 'added frame {0}'.format(frame_no)
        frame_no+=1


out_img = drizzle.output(drizzled_img,weight_img)

out_img = np.uint16(np.round(out_img))

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)
cv2.imwrite(outname,out_img)

