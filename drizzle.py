#!/usr/bin/env python

# some example transforms
import numpy as np
import cv2
import math
import sys
import _drizzle
offsets = np.array([(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)])

# sort corners in clockwise direction, taken from
# http://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
def order_vertices(corners):
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of corners which includes angles
    cornersWithAngles = []
    for x, y in corners:
        dx = x - cx
        dy = y - cy
        an = (math.atan2(dx, dy) + 2.0 * math.pi) % (2.0 * math.pi)
        cornersWithAngles.append((dx, dy, an))
    # sort it using the angles
    cornersWithAngles.sort(key = lambda tup: tup[2], reverse=True)
    return [(x,y) for (x,y,theta) in cornersWithAngles]


# taken from
# http://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
def shoelace_area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# determines if vertex is 'inside' line for clockwise polygon
def is_inside(line,vertex):
    Ax = line[0,0]
    Ay = -line[0,1]

    Bx = line[1,0]
    By = -line[1,1]

    Px = vertex[0]
    Py = -vertex[1]

    angle =  ((Bx-Ax)*(Py-Ay) - (By-Ay)*(Px-Ax))
    
    return angle <= 0


# returns coordinates of intersection of two lines
# based on https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
# I know, its insane.
def intersection(l1,l2):
    invert_y = np.array(((1,-1),(1,-1)),dtype=np.float32)
    l1 = l1*invert_y
    l2 = l2*invert_y

    d = ((l1[0,0]-l1[1,0])*(l2[0,1]-l2[1,1])-(l1[0,1]-l1[1,1])*(l2[0,0]-l2[1,0]))
    x = ((l1[0,0]*l1[1,1]-l1[0,1]*l1[1,0])*(l2[0,0]-l2[1,0])-(l1[0,0]-l1[1,0])*(l2[0,0]*l2[1,1] - l2[0,1]*l2[1,0]))/d
    
    y = ((l1[0,0]*l1[1,1]-l1[0,1]*l1[1,0])*(l2[0,1]-l2[1,1])-(l1[0,1]-l1[1,1])*(l2[0,0]*l2[1,1] - l2[0,1]*l2[1,0]))/d

    return (x,-y)

# compute gift wrapping algo, so we can determine if quads are convex
def jarvis(S):
    # get the leftmost point
    pointOnHull = min(list(S), key=lambda p: p[0])
    P = [None] * len(S)
    endpoint = np.nan
    i = 0
    while True:
        P[i] = pointOnHull
        endpoint = S[0]
        for j in range(1,len(S)):
            line =  np.array((P[i],endpoint))
            if((endpoint == pointOnHull).all() or not is_inside(line, S[j])):
                endpoint = S[j]
        i += 1
        pointOnHull = endpoint
        if (endpoint == P[0]).all():
            break;
    return i
    

# returns intersection of clip and subject polygons 
def sutherland_hodgman(clip,subject):
    c_points = len(clip)

    # assert that inputs are convex 
    if jarvis(clip) != len(clip): return None
    if jarvis(subject) != len(subject): return None
        
    candidates = []
    for c0 in range(c_points):
        c1 = (c0 + 1) % c_points
        line = np.array((clip[c0],clip[c1]))
        s_points = len(subject)
        
        poly=[]
        for s0 in range(s_points):
            s1 = (s0 + 1) % s_points
            s0 = subject[s0]
            s1 = subject[s1]

            if is_inside(line,s0):
                if is_inside(line,s1):
                    poly.append(s1)
                else:
                    p = intersection(line,np.array((s0,s1)))
                    poly.append(p)
            else:
                if is_inside(line,s1):
                    p = intersection(line,np.array((s0,s1)))
                    poly.append(p)
                    poly.append(s1)
        subject = poly
        candidates.append(len(poly))
    return  np.array(poly)


def get_quad_corners(x,y,M,pixfrac,scalefrac):
    quad_corners = np.empty(offsets.shape)
    # get the coordinates of the point in the destination image 
    for i in range(4):
        quad_corners[i] = get_coords(pixfrac*offsets[i]+(x,y),M) / scalefrac
    return quad_corners


# no longer a class
# takes several parameters:
# src: source image to drizzle from
# dest: dest image for data
# M: perspective transform
# src_weights: weights for each pixel of src
# pixfrac: pixfrac
# scalefrac: scalefrac

def drizzle(src, dest, M, dest_weight, src_weights=None, pixfrac=0.5, scalefrac=0.4):
    """Drizzle a given image with a given perspective transform onto the destination image"""
    rows, cols, depth = src.shape
    drows, dcols, ddepth = dest.shape

    dest_corners = np.empty(offsets.shape)
    quad_corners = np.zeros(offsets.shape)
    if not src_weights.any():
        src_weights = np.ones(src.shape)

    # for every pixel in the image
    sys.stdout.write('processing row')
    for py in np.arange(rows):
        if (py % 100) == 0:
            sys.stdout.write(' {0}'.format(py))
            sys.stdout.flush()
        for px in np.arange(cols):
            quad_corners = get_quad_corners(px,py,M,pixfrac,scalefrac)

            # get the total area of this quad
            quad_area = shoelace_area(quad_corners)

            pixweight = src_weights[py,px]
            pixval = src[py,px]

            cy = [y for (x,y) in quad_corners]
            cx = [x for (x,y) in quad_corners]
            # get the bounding box of the quad
            minx = int(np.round(np.clip(np.min(cx),0,dcols)))
            maxx = int(np.round(np.clip(np.max(cx),0,dcols)))
            miny = int(np.round(np.clip(np.min(cy),0,drows)))
            maxy = int(np.round(np.clip(np.max(cy),0,drows)))

            dbox = np.array([[0,0],[dcols,0],[dcols,drows],[0,drows]])

            if not sutherland_hodgman(dbox,quad_corners).any():
                continue

            # for every pixel in the bounding box
            for i in range(minx,maxx):
                for j in range(miny,maxy):

                    dest_corners = np.repeat([[i,j]],4,axis=0) + offsets

                    # get the polygon that represents intersection
                    poly = sutherland_hodgman(quad_corners, dest_corners)
                    # done, if there is nothing more for this pixel
                    if not poly.any():
                        continue
                    poly_area = shoelace_area(poly)

                    weight = pixweight * poly_area / quad_area

                    dest[j,i] += weight * pixval
                    dest_weight[j,i] += weight
    print ' '

# dest_img is output
def output(driz_img,weight_img):
    weight = np.copy(weight_img)
    weight[weight == 0] = 1

    out = driz_img / weight
    out[out < 0] = 0
    return out

def build_dest_array(src,pixfrac=0.5,scalefrac=0.4):
    assert pixfrac >= 0 and pixfrac <= 1
    assert scalefrac > 0 and scalefrac <= 1
    rows,cols,depth = src.shape
    return np.zeros((rows/scalefrac, cols/scalefrac,depth), dtype=np.float32)
    
def get_coords(point, M):
    (x,y) = point
        
    x_ = (M[0,0]*x + M[0,1]*y + M[0,2])/(M[2,0]*x + M[2,1]*y + M[2,2])
    y_ = (M[1,0]*x + M[1,1]*y + M[1,2])/(M[2,0]*x + M[2,1]*y + M[2,2])
    return np.array((x_,y_))
