#ifndef DRIZZLE_H__
#define DRIZZLE_H__

#define ALG_DRIZZLE 0
#define ALG_LANCZOS3 1
#define ALG_INTERLACE 2

/* def drizzle(src, dest, M, dest_weight, src_weights=None, pixfrac=0.5, scalefrac=0.4): */
int drizzle(const float * src, const long * src_shape,
	    float * dest, const long * dest_shape,
	    const float * M,
	    float * dest_weight,
	    const float * src_weights,
	    float pixfrac, float scalefrac);

int interlace(const float * src, const long * src_shape,
	      float * dest, const long * dest_shape,
	      const float * M,
	      float * dest_weight,
	      const float * src_weights, float scalefrac);

int lanczos3(const float * src, const long * src_shape,
	     float * dest, const long * dest_shape,
	     const float * M,
	     float * dest_weight,
	     const float * src_weights, float scalefrac);


#endif	/* DRIZZLE_H__ */
