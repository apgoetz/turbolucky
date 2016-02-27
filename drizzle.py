#!/usr/bin/env python

# some example transforms
import numpy as np
import cv2
import math
import sys

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
    
    return angle<= 0


# returns coordinates of intersection of two lines
# based on https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
# I know, its insane

def intersection(l1,l2):
    x = ((l1[0,0]*l1[1,1]-l1[0,1]*l1[1,0])*(l2[0,0]-l2[1,0])-(l1[0,0]-l1[1,0])*(l2[0,0]*l2[1,1] - l2[0,1]*l2[1,0]))/((l1[0,0]-l1[1,0])*(l2[0,1]-l2[1,1])-(l1[0,1]-l1[1,1])*(l2[0,0]-l2[1,0]))

    y = ((l1[0,0]*l1[1,1]-l1[0,1]*l1[1,0])*(l2[0,1]-l2[1,1])-(l1[0,1]-l1[1,1])*(l2[0,0]*l2[1,1] - l2[0,1]*l2[1,0]))/((l1[0,0]-l1[1,0])*(l2[0,1]-l2[1,1])-(l1[0,1]-l1[1,1])*(l2[0,0]-l2[1,0]))

    return (x,y)
    
# returns intersection of clip and subject polygons 
def sutherland_hodgman(clip,subject):
    c_points = len(clip)

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

    return np.array(poly)

class Drizzle:
    offsets = np.array([(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)])
    
    def __init__(self, pixfrac, sourceshape, scalefrac):
        assert pixfrac >= 0 and pixfrac <= 1
        assert scalefrac > 0 and scalefrac <= 1
        self.pixfrac = pixfrac
        self.scalefrac = scalefrac
        self.sourceshape = sourceshape
        rows,cols,depth = sourceshape
        self.dest_img = np.zeros((rows/scalefrac, cols/scalefrac,depth))
        self.weight_img = np.zeros(self.dest_img.shape)

    def get_quad_corners(self,x,y,M):
        quad_corners = np.empty(Drizzle.offsets.shape)
        # get the coordinates of the point in the destination image 
        for i in range(4):
            quad_corners[i] = get_coords(self.pixfrac*Drizzle.offsets[i]+(x,y),M) / self.scalefrac
        return quad_corners

    def drizzle(self,img, M, weights=None):
        """Drizzle a given image with a given perspective transform onto the destination image"""
        rows, cols, depth = img.shape
        drows, dcols, ddepth = self.dest_img.shape

        dest_corners = np.empty(Drizzle.offsets.shape)
        quad_corners = np.zeros(Drizzle.offsets.shape)
        if not weights.any():
            weights = np.ones(img.shape)
        
        # for every pixel in the image
        sys.stdout.write('processing column')
        for px in np.arange(cols):
            if (px % 100) == 0:
                sys.stdout.write(' {0}'.format(px))
                sys.stdout.flush()
            for py in np.arange(rows):
                quad_corners = self.get_quad_corners(px,py,M)

                # get the total area of this quad
                quad_area = shoelace_area(quad_corners)

                pixweight = weights[py,px]
                pixval = img[py,px]
                
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

                        dest_corners = np.repeat([[i,j]],4,axis=0) + Drizzle.offsets

                        # get the polygon that represents intersection
                        poly = sutherland_hodgman(quad_corners, dest_corners)
                        # done, if there is nothing more for this pixel
                        if not poly.any():
                            continue
                        poly_area = shoelace_area(poly)

                        weight = pixweight * poly_area / quad_area

                        self.dest_img[j,i] += weight * pixval
                        self.weight_img[j,i] += weight
        print ' '
    def output(self):
        weight = np.copy(self.weight_img)
        weight[weight == 0] = 1
        return self.dest_img / weight

def get_coords(point, M):
    (x,y) = point
        
    x_ = (M[0,0]*x + M[0,1]*y + M[0,2])/(M[2,0]*x + M[2,1]*y + M[2,2])
    y_ = (M[1,0]*x + M[1,1]*y + M[1,2])/(M[2,0]*x + M[2,1]*y + M[2,2])
    return np.array((x_,y_))



# test

#img = cv2.imread('image.png')

# M = np.array([[  1.00271698e+00,   2.31601371e-03,   7.48436820e+00],[  1.50137800e-03,   1.00314667e+00,  -7.73453684e+00],[  8.11759411e-07,   1.56448983e-06,   1.00000000e+00]])

# driz = Drizzle(0.5,img.shape,0.4)
# driz.drizzle(img,M)
        
# cv2.imwrite('out.tif', driz.output())
