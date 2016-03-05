#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

bool DO_LOG = false;

#define log(...) if (DO_LOG) {printf(__VA_ARGS__);}

/* make a line from two points */
static void mkline(float line[2][2], const float p1[2], const float p2[2])
{
	int i;
	for (i = 0; i < 2; i++)
	{
		line[0][i] = p1[i];
		line[1][i] = p2[i];
	}
}

/* copy a point */
static void pt_cpy(float dest[2], const float src[2])
{
	int i;
	for (i = 0; i < 2; i++) {
		dest[i] = src[i];
	}
}

/* compare two points to see if they are the same  */
static bool pt_cmp(const float p1[2], const float p2[2])
{
	int i;
	for (i = 0; i < 2; i++) {
		if (p1[i] != p2[i]) {
			return false;
		}
	}
	return true;
}


/* return clipped value */
static float clip(float val, float min, float max)
{
	if (val < min) {
		return min;
	} else if (val > max){
		return max;
	} else {
		return val;
	}
}

/* return min value in array */
float min_arr(float * vec, size_t cnt)
{
	size_t i;
	float min = vec[0];
	for (i = 0; i < cnt; i++) {
		if (vec[i] < min) {
			min = vec[i];
		}
	}
	return min;
}

/* return max value in array */
float max_arr(float * vec, size_t cnt)
{
	size_t i;
	float max = vec[0];
	for (i = 0; i < cnt; i++) {
		if (vec[i] > max) {
			max = vec[i];
		}
	}
	return max;
}



/* static funcs */
/* def get_quad_corners(x,y,M,pixfrac,scalefrac): */

/* transforms coordinates x and y by matrix M */
/* x and y are coords
 * corners must by 4x2 matrix
 *
 * python signature: 
 * get_quad_corners(x,y,M,pixfrac,scalefrac)
 */
static void get_quad_corners(float px, float py,
			     const float M[3][3], float pixfrac,
			     float scalefrac, float corners[4][2])
{

/* the offsets that are added to get clockwise quad */
	const float offset_matrix[4][2]  = {{-.5,-.5},{.5,-.5},{.5,.5},{-.5,.5}};
	
	
	int i;
	float x_, y_,x,y;

	/* for every point of the quad */
	for (i = 0; i < 4; i++) {
		/* get corners of quad */
		x = offset_matrix[i][0]*pixfrac + px;
		y = offset_matrix[i][1]*pixfrac + py;
				
		/* transform by M */
		x_ = (M[0][0]*x + M[0][1]*y + M[0][2])/(M[2][0]*x + M[2][1]*y + M[2][2]);
		y_ = (M[1][0]*x + M[1][1]*y + M[1][2])/(M[2][0]*x + M[2][1]*y + M[2][2]);

		/* scale by output image size */
		corners[i][0] = x_ / scalefrac;
		corners[i][1] = y_ / scalefrac;
	}
}

/* gets the intersection between the lines l1 and l2 as a point
 *
 * based on https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
 * 
 * point must be a 2 vector
 */
static void get_intersection(const float l1[2][2], const float l2[2][2], float * point)
{
	/* need to convert from 'cv' coordinates to cartesian coordinates */
	double l1c[2][2];
	double l2c[2][2];
	int i;
	
	for (i = 0; i < 2; i++) {
		l1c[i][0] = l1[i][0];
		l1c[i][1] = -l1[i][1];
		l2c[i][0] = l2[i][0];
		l2c[i][1] = -l2[i][1];
	}
	

	double denom = ((l1c[0][0]-l1c[1][0])*(l2c[0][1]-l2c[1][1])-(l1c[0][1]-l1c[1][1])*(l2c[0][0]-l2c[1][0]));
	
	double x = ((l1c[0][0]*l1c[1][1]-l1c[0][1]*l1c[1][0])*(l2c[0][0]-l2c[1][0])-(l1c[0][0]-l1c[1][0])*(l2c[0][0]*l2c[1][1] - l2c[0][1]*l2c[1][0]))/denom;

	double y = ((l1c[0][0]*l1c[1][1]-l1c[0][1]*l1c[1][0])*(l2c[0][1]-l2c[1][1])-(l1c[0][1]-l1c[1][1])*(l2c[0][0]*l2c[1][1] - l2c[0][1]*l2c[1][0]))/denom;

	point[0] = (float)x;
	point[1] = (float)-y;		/* convert back to cv coordinates */

}

static void matmul(const float M[2][2], const float v[2], float out[2])
{
	int i;
	
	for (i = 0; i < 2; i++) {
		out[i] = M[i][0] * v[i] + M[i][1] * v[i];
	}
}

static void parametric_intersection(const float l1[2][2], const float l2[2][2], float * point)
{
	float l1p[2][2];
	float l2p[2][2];
	int i;

	float Ax = l1[0][0];
	float Ay = -l1[0][1];
	
	/* copy over points, and shift their values by A */
	for (i = 0; i < 2; i++) {
		l1p[i][0] = l1[i][0] - Ax;
		l1p[i][1] = -l1[i][1] - Ay;

		
		l2p[i][0] = l2[i][0] - Ax;
		l2p[i][1] = -l2[i][1] - Ay;
	}

	
	
	float Ap[2];
	float Bp[2];
	float Cp[2];
	float Dp[2];

	pt_cpy(&Ap[0], &l1p[0][0]);
	pt_cpy(&Bp[0], &l1p[1][0]);
	pt_cpy(&Cp[0], &l2p[0][0]);
	pt_cpy(&Dp[0], &l2p[1][0]);
	
	float L = sqrtf(Bp[0]*Bp[0] - Bp[1]*Bp[1]);

	float sin_theta = Bp[0] / L;
	float cos_theta = Bp[1] / L;

	float R[2][2] = {{cos_theta, -sin_theta},{sin_theta, cos_theta}};
	float Rinv[2][2] = {{cos_theta, sin_theta},{-sin_theta, cos_theta}};

	float Cr[2];
	float Dr[2];

	matmul(R,Cp, Cr);
	matmul(R,Dp, Dr);

	float Pr[2] = {Dr[0] - Dr[1]*(Cr[0] - Dr[0])/(Cr[1] - Dr[1]), 0};
	float Pp[2];

	matmul(Rinv, Pr, Pp);

	point[0] += Ax;
	point[1] += Ay;

	point[1] *= -1;
}

/* uses cross product to determine if vertex is inside line, assuming
 * line comes from a clockwise polygon
 *
 * line is a 2x2 matrix
 * vertex is a 2 vector
 *
 * returns true if point is inside, false otherwise
 *
 * python: is_inside(line, vertex)
*/
static bool is_inside(const float  line[2][2], const float vertex[2])
{
	float Ax,Ay,Bx,By,Px,Py;

	Ax =  line[0][0];
	Ay = -line[0][1];

	Bx =  line[1][0];
	By = -line[1][1];
	
	Px =  vertex[0];
	Py = -vertex[1];
	
	float  angle = ((Bx-Ax)*(Py-Ay) - (By-Ay)*(Px-Ax));
	
	return angle <= 0;
}


/* Calculate the number of points in the convex hull of the passed in
 * poly. Used to determine if a poly is convex or not */
static size_t jarvis(const float S[][2], size_t n_S)
{
	/* place to put elements into */
	float P[n_S][2];
	float pointOnHull[2];
	float endpoint[2];
	float line[2][2];
	size_t i;
	size_t j;

	/* set pointonhull to be leftmost point */
	pt_cpy(pointOnHull, &S[0][0]);
	for (i = 0; i < n_S; i++) {
		if (S[i][0] < pointOnHull[0]) {
			pt_cpy(pointOnHull, &S[i][0]);			
		}
	}

	i = 0;
	do {
		pt_cpy(&P[i][0],pointOnHull);
		pt_cpy(endpoint, &S[0][0]);
		
		for (j = 1; j < n_S; j++) {

			pt_cpy(&line[0][0], &P[i][0]);
			pt_cpy(&line[1][0], endpoint);
			
			if (pt_cmp(endpoint, pointOnHull) || !is_inside(line,&S[j][0])) {
				pt_cpy(endpoint, &S[j][0]);
			}
		}
		i++;
		pt_cpy(pointOnHull, endpoint);
	} while(!pt_cmp(endpoint, &P[0][0]));
	return i;
}

/* performs sutherland hodgman from clip poly to subject poly
 * note: all coordinates must be oriented clockwise
 *
 * clip is a clip_pointsx2 matrix
 * subject is a subject_pointsx2 matrix
 * result is placed in output_coords
 * output_coords must by 2*subject_pointsx2 matrix
 *
 * returns number of coords in output_coords, or 0 for none
 *
 * python:
 * sutherland_hodgman(clip,subject)
 */
static long sutherland_hodgman(const float clip[][2], size_t clip_points,
			      const float subject[][2], size_t subject_points,
			      float output_coords[][2])
{
	/* number of output coordinates */
	size_t n_out, n_candidate;
	size_t c0,c1,s0,s1;
	float s0_pt[2];
	float s1_pt[2];
	float intersect_pt[2];
	size_t i;
	float line[2][2];
	float subj_line[2][2];
	float  candidate_points[2*subject_points][2];



	/* if clip is concave, say area was zero */
	if (jarvis(clip, clip_points) != clip_points) {
		log("nonconvex clip!\n");
		for (i = 0; i < clip_points; i++)
		{
			log("(%f,%f)\n", clip[i][0], clip[i][1]);
		}
		
		return 0;
	}

	/* if subject was concave, say area was zero */
	if (jarvis(subject, subject_points) != subject_points) {
		log("nonconvex subject!\n");
		return 0;
	}

	n_out = subject_points;
	
	/* copy over the subject points to the output list. This is
	 * the starting point to look at */
	memcpy(output_coords,subject, 2*sizeof(float)*n_out);

	for (c0 = 0; c0 < clip_points; c0++) {
		c1 = (c0+1) % clip_points;

		/* make the line of the clip poly we will use this iteration */
		mkline(line,&clip[c0][0], &clip[c1][0]);
		
		/* copy over whatever said was the output last time */
		memcpy(candidate_points,output_coords, 2*sizeof(float)*n_out);
		n_candidate = n_out;
		
		/* reset the number of out for this loop */
		n_out = 0;
	
		for (s0 = 0; s0 < n_candidate; s0++) {
			s1 = (s0 + 1) % n_candidate;
			/* get the vectors for the points */
			pt_cpy(s0_pt, &candidate_points[s0][0]);
			pt_cpy(s1_pt, &candidate_points[s1][0]);

			if (is_inside(line,s0_pt)) {
				if (is_inside(line,s1_pt)) {
					/* append pt1 */
					pt_cpy(&output_coords[n_out][0], s1_pt);
					n_out++;
				} else {

					mkline(subj_line, s0_pt, s1_pt);
					get_intersection(line,subj_line,intersect_pt);
					pt_cpy(&output_coords[n_out][0], intersect_pt);
					n_out++;
				}
			} else {
				if (is_inside(line,s1_pt)) {
					mkline(subj_line, s0_pt, s1_pt);
					get_intersection(line,subj_line,intersect_pt);
					pt_cpy(&output_coords[n_out][0],intersect_pt);
					n_out++;

					pt_cpy(&output_coords[n_out][0], s1_pt);
					n_out++;
				}
			}
		}
		
	}
	return n_out;
}

/*
 * Calculate area of clockwise convex polygon using shoelace formula
 *
 * corners is a n_cornersx2 matrix  
 * returns the area enclosed
 *
 * python: shoelace_area(corners)
 */
static float shoelace_area(const float  corners[][2], size_t n_corners)
{
	float area = 0;
	size_t i,j;
	float centroid[2] = {0,0};
	float line[2][2];
	bool last_inside;
	for (i = 0; i < n_corners; i++) {
		centroid[0] += corners[i][0];
		centroid[1] += corners[i][1];
	}

	centroid[0] /= n_corners;
	centroid[1] /= n_corners;
		
	for (i = 0; i < n_corners; i++) {
		j = (i + 1) % n_corners;
		line[0][0] = corners[i][0];
		line[0][1] = corners[i][1];
		line[1][0] = corners[j][0];
		line[1][1] = corners[j][1];

		if (i == 0) {
			last_inside = is_inside(line,centroid);
		} else if (is_inside(line,centroid) != last_inside) {
			log("uhoh! %ld\n", n_corners);
			for (i = 0; i < n_corners; i++) {
				log("(%f,%f)\n", corners[i][0], corners[i][1]);
			}
			return -1;
		} 
		
		area += corners[i][0] * corners[j][1];
		area -= corners[j][0] * corners[i][1];
	}
	
	area = fabsf(area) / 2;
	return area;
}


/* array shape: rows, then cols, then depth */
/* 3 planes are next to each other, then cols move faster, rows are slowest */
/* accessing an element:  */
/* image [i,j,k] = image[i*depth*cols + j*depth + k] */



/* def drizzle(src, dest, M, dest_weight, src_weights=None, pixfrac=0.5, scalefrac=0.4): */
int drizzle(const float * src, const long * src_shape,
	    float * dest, const long * dest_shape,
	    const float M[3][3],
	    float * dest_weight,
	    const float * src_weights,
	    float pixfrac, float scalefrac)
{
	long rows, cols, depth;
	long drows, dcols;
	long px,py;
	long i,j,k;
	
	rows = src_shape[0];
	cols = src_shape[1];
	depth = src_shape[2];
	drows = dest_shape[0];
	dcols = dest_shape[1];
	float quad_corners[4][2];

	float quad_area, poly_area;
	float outpoly[8][2];

	/* it works! */
	for (py = 0; py < rows; py++) {

	
		for (px = 0; px < cols; px++) {

			get_quad_corners(px,py,M,pixfrac,scalefrac, quad_corners);
			
			quad_area = shoelace_area(quad_corners, 4);
			if (quad_area < 0) {
				quad_area = 0;
			}
			
			float pixweight[depth];
			float pixval[depth];			
			for (i = 0; i < depth; i++){
				pixweight[i] = src_weights[py*depth*cols + px*depth + i];
				pixval[i] = src[py*depth*cols + px*depth + i];
			}

			float cy[4];
			float cx[4];
			
			for (i = 0; i < 4; i++) {
				cx[i] = quad_corners[i][0];
				cy[i] = quad_corners[i][1];
				
	
			}
	
			long minx,maxx,miny,maxy;

			/* todo, handle case where inside one pixel */
			minx = lroundf(clip(min_arr(cx,4)-1,0,dcols));
			maxx = lroundf(clip(max_arr(cx,4)+1,0,dcols));
			miny = lroundf(clip(min_arr(cy,4)-1,0,drows));
			maxy = lroundf(clip(max_arr(cy,4)+1,0,drows));

			
			float dbox [4][2] = {{0,0},{dcols,0},{dcols,drows},{0,drows}};

			long out_pts;
			
			/* check if the quad is on the output image at all */
			if (sutherland_hodgman(dbox, 4, quad_corners, 4, outpoly) == 0) {
				continue;
			}
			
			/* for each pixel in the affected region */
			for (i = minx; i < maxx; i++) {
				for (j = miny; j < maxy; j++) {
					float dest_corners[4][2];
					const float eye3[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

					/* get the coordinates of the
					 * output pixel */
					get_quad_corners(i,j,eye3, 1,1, dest_corners);
					
					/* find intersection of pixel and quad */
					out_pts = sutherland_hodgman(quad_corners, 4,
								     dest_corners, 4,
								     outpoly);



					/* if no overlap, skip */
					if (out_pts == 0) {
						continue;
					} else if (out_pts < 0) {
					}					

					
					if(out_pts < 3 || out_pts > 8) {
						log("oops: %ld pts in overlap\n", out_pts);
						log("quad:\n");
						for (i = 0; i < 4; i++) {
							log("(%f,%f)\n",quad_corners[i][0],
							       quad_corners[i][1]);
						}

						log("dest:\n");
						for (i = 0; i < 4; i++) {
							log("(%f,%f)\n",dest_corners[i][0],
							       dest_corners[i][1]);
						}
	
						return -1;
	
	
					}

					/* calculate area of overlap */
					poly_area = shoelace_area(outpoly, out_pts);

					if (poly_area < 0) {
						poly_area = 0;
						log("quad:\n");
						for (i = 0; i < 4; i++) {
							log("(%f,%f)\n",quad_corners[i][0],
							       quad_corners[i][1]);
						}

						log("dest:\n");
						for (i = 0; i < 4; i++) {
							log("(%f,%f)\n",dest_corners[i][0],
							       dest_corners[i][1]);
						}
					}

					float weight;
	
					for (k = 0; k < depth; k++) {
						if (quad_area == 0){
							continue;
						}

						weight = pixweight[k] * poly_area / quad_area;

						dest[j*depth*dcols + i*depth + k] += weight * pixval[k];
						dest_weight[j*depth*dcols + i*depth + k] += weight;
						
					}
					

				}
			}
		}
		
	}
	return 1;
}
