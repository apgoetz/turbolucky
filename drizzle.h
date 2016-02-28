#ifndef DRIZZLE_H__
#define DRIZZLE_H__


/* def drizzle(src, dest, M, dest_weight, src_weights=None, pixfrac=0.5, scalefrac=0.4): */
int drizzle(const float * src, const long * src_shape,
	    float * dest, const long * dest_shape,
	    const float * M,
	    float * dest_weight,
	    const float * src_weights,
	    float pixfrac, float scalefrac);


#endif	/* DRIZZLE_H__ */
