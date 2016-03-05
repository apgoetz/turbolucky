/* Python C  */
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "drizzle.h"
#include "stdbool.h"

static char module_docstring[] =
    "This module provides an interface for drizzling images.";
static char drizzle_docstring[] =
    "Drizzle an image on a dest image.";

static PyObject *drizzle_drizzle(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"drizzle", drizzle_drizzle, METH_VARARGS, drizzle_docstring},
    {NULL, NULL, 0, NULL}
};



PyMODINIT_FUNC init_drizzle(void)
{
    PyObject *m = Py_InitModule3("_drizzle", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* add the options for the algorithm choice */
    PyModule_AddIntConstant(m, "ALG_DRIZZLE", ALG_DRIZZLE);
    PyModule_AddIntConstant(m, "ALG_LANCZOS3", ALG_LANCZOS3);
    PyModule_AddIntConstant(m, "ALG_INTERLACE", ALG_INTERLACE);

    /* Load `numpy` functionality. */
    import_array();
}


/* check if elements of array are equal */
static bool float_long_eql(long * a, long * b, size_t size)
{
	size_t i;
	for (i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

/* def drizzle(src, dest, M, dest_weight, src_weights=None, pixfrac=0.5, scalefrac=0.4): */

static PyObject *drizzle_drizzle(PyObject *self, PyObject *args)
{
	double pixfrac, scalefrac;
	PyObject *src_obj, *dest_obj, *M_obj, *dest_weight_obj, *src_weights_obj;
	int ret = -1;
	int algorithm;
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OO!OO!Oddi", &src_obj, &PyArray_Type, &dest_obj,
			      &M_obj, &PyArray_Type, &dest_weight_obj,
			      &src_weights_obj, &pixfrac, &scalefrac, &algorithm))
		return NULL;

	
	PyArrayObject *src_array = (PyArrayObject*)PyArray_FROM_OTF(src_obj, NPY_FLOAT32,  NPY_ARRAY_IN_ARRAY);
	PyArrayObject *dest_array = (PyArrayObject*)PyArray_FROM_OTF(dest_obj, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
	PyArrayObject *M_array = (PyArrayObject*)PyArray_FROM_OTF(M_obj, NPY_FLOAT32,  NPY_ARRAY_IN_ARRAY);
	PyArrayObject *dest_weight_array = (PyArrayObject*)PyArray_FROM_OTF(dest_weight_obj, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
	PyArrayObject *src_weights_array = (PyArrayObject*)PyArray_FROM_OTF(src_weights_obj, NPY_FLOAT32,  NPY_ARRAY_IN_ARRAY);

	/* If that didn't work, throw an exception. */
	if (src_array == NULL || dest_array == NULL || M_array == NULL
	    || dest_weight_array == NULL || src_weights_array == NULL ) {
		
		goto cleanup;
    }

	

	/* now we verify the shape of the arrays */
	if (PyArray_NDIM(src_array) != 3) {
		PyErr_SetString(PyExc_RuntimeError, "Src has wrong shape!");
		goto cleanup;
	}

	if (PyArray_NDIM(dest_array) != 3) {
		PyErr_SetString(PyExc_RuntimeError, "dest has wrong shape!");
		goto cleanup;
	}

	if (PyArray_NDIM(M_array) != 2) {
		PyErr_SetString(PyExc_RuntimeError, "M has wrong shape!");
		goto cleanup;
	}

	if (PyArray_NDIM(dest_weight_array) != 3) {
		PyErr_SetString(PyExc_RuntimeError, "dest_weight has wrong shape!");
		goto cleanup;
	}

	if (PyArray_NDIM(src_weights_array) != 3) {
		PyErr_SetString(PyExc_RuntimeError, "src_weights has wrong shape!");
		goto cleanup;
	}
	
	if (! float_long_eql(PyArray_DIMS(src_array), PyArray_DIMS(src_weights_array), 3)) {
		PyErr_SetString(PyExc_RuntimeError, "src size doesn't match src_weights size!");
		goto cleanup;
	}

	if (! float_long_eql(PyArray_DIMS(dest_array), PyArray_DIMS(dest_weight_array), 3)) {
		PyErr_SetString(PyExc_RuntimeError, "dest size doesn't match dest_weight size!");
		goto cleanup;
	}

	long * M_size = PyArray_DIMS(M_array);
	if (M_size[0] != 3 || M_size[1] != 3) {
		PyErr_SetString(PyExc_RuntimeError, "M has wrong size!");
		goto cleanup;
	}

	long * src_shape = PyArray_DIMS(src_array);
	long * dest_shape = PyArray_DIMS(dest_array);


	float *src = (float*)PyArray_DATA(src_array);
	float *dest = (float*)PyArray_DATA(dest_array);
	float *M = (float*)PyArray_DATA(M_array);
	float *dest_weight = (float*)PyArray_DATA(dest_weight_array);
	float *src_weights = (float*)PyArray_DATA(src_weights_array);

	/* finally, call it! */
	switch (algorithm) {
	case ALG_DRIZZLE:
		ret = drizzle(src,src_shape,
			      dest,dest_shape,
			      M,
			      dest_weight,
			      src_weights,
			      pixfrac,scalefrac);

		break;
	case ALG_LANCZOS3:
		ret = lanczos3(src,src_shape,
			      dest,dest_shape,
			      M,
			      dest_weight,
			      src_weights,
			      scalefrac);

		break;
	case ALG_INTERLACE:
		ret = interlace(src,src_shape,
			      dest,dest_shape,
			      M,
			      dest_weight,
			      src_weights,
			      scalefrac);

		break;
	default:
		PyErr_SetString(PyExc_RuntimeError, "Unknown Algorithm!");
	}
	
	
	
cleanup:
	Py_XDECREF(src_array);
	Py_XDECREF(dest_array);
	Py_XDECREF(M_array);
	Py_XDECREF(dest_weight_array);
	Py_XDECREF(src_weights_array);
	
	if (ret >= 0) {
		return Py_BuildValue("i", ret);
	} else {
		return NULL;
	}
}
