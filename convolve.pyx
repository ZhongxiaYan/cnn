import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
# def convolve(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[DTYPE_t, ndim=2] g):
#     if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
#         raise ValueError("Only odd dimensions on filter supported")
#     assert f.dtype == DTYPE and g.dtype == DTYPE
#     # The "cdef" keyword is also used within functions to type variables. It
#     # can only be used at the top indentation level (there are non-trivial
#     # problems with allowing them in other places, though we'd love to see
#     # good and thought out proposals for it).
#     #
#     # For the indices, the "int" type is used. This corresponds to a C int,
#     # other C types (like "unsigned int") could have been used instead.
#     # Purists could use "Py_ssize_t" which is the proper Python type for
#     # array indices.
#     cdef int vmax = f.shape[0]
#     cdef int wmax = f.shape[1]
#     cdef int smax = g.shape[0]
#     cdef int tmax = g.shape[1]
#     cdef int smid = smax // 2
#     cdef int tmid = tmax // 2
#     cdef int xmax = vmax + 2*smid
#     cdef int ymax = wmax + 2*tmid
#     cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros([xmax, ymax], dtype=DTYPE)
#     cdef int x, y, s, t, v, w
#     # It is very important to type ALL your variables. You do not get any
#     # warnings if not, only much slower code (they are implicitly typed as
#     # Python objects).
#     cdef int s_from, s_to, t_from, t_to
#     # For the value variable, we want to use the same data type as is
#     # stored in the array, so we use "DTYPE_t" as defined above.
#     # NB! An important side-effect of this is that if "value" overflows its
#     # datatype size, it will simply wrap around like in C, rather than raise
#     # an error like in Python.
#     cdef DTYPE_t value
#     for x in range(xmax):
#         for y in range(ymax):
#             s_from = max(smid - x, -smid)
#             s_to = min((xmax - x) - smid, smid + 1)
#             t_from = max(tmid - y, -tmid)
#             t_to = min((ymax - y) - tmid, tmid + 1)
#             value = 0
#             for s in range(s_from, s_to):
#                 for t in range(t_from, t_to):
#                     v = x - smid + s
#                     w = y - tmid + t
#                     value += g[smid - s, tmid - t] * f[v, w]
#             h[x, y] = value
#     return h

# def convForward(np.ndarray[np.float_t, ndim=2] batch, np.ndarray[np.float_t, ndim=2] filt, int dim_image, int dim_filter, int num_filters):
#     '''
#     batch: <batch_size> * <dim_image ^ 2>
#     filt: <num_filters> * <dim_filter ^ 2>
#     output: 2d convolution of each row in batch with filt, for a total of <batch_size> rows
#     Assume stride = 1, filt_dim = 5, dim_image = 28
#     '''

#     cdef int zero_pad = 1 # padding on each side
#     cdef int stride = 1
#     cdef int output_dim = (dim_image - dim_filter + 2 * zero_pad) // stride + 1
#     cdef np.ndarray[np.float_t, ndim=2] result = np.zeros((batch.shape[0], output_dim * output_dim * num_filters), dtype=np.float)

def convolve2D(np.ndarray[np.float_t, ndim=1] a_1, np.ndarray[np.float_t, ndim=1] a_2, int dim_a_1, int dim_a_2, int padding, int stride, np.ndarray[np.float_t, ndim=1] output):
    """
    padding: padding on left, right, top, and bottom
    'valid' convolution: dim_a_2 <= dim_a_1
    """
    cdef int dim_output = (dim_a_1 - dim_a_2 + 2 * padding) // stride + 1
    cdef int x_out, y_out
    cdef int x_1, y_1, x_2, y_2, x_1_2, y_1_2
    cdef np.float_t output_xy
    for x_out in range(dim_output):
        x_1 = x_out * stride - padding
        for y_out in range(dim_output):
            y_1 = y_out * stride - padding
            output_xy = 0
            for x_2 in range(dim_a_2):
                for y_2 in range(dim_a_2):
                    y_1_2 = (y_1 + y_2)
                    x_1_2 = (x_1 + x_2)
                    if y_1_2 < 0 or x_1_2 < 0 or y_1_2 >= dim_a_1 or x_1_2 >= dim_a_1:
                        continue
                    output_xy += a_1[y_1_2 * dim_a_1 + x_1_2] * a_2[y_2 * dim_a_2 + x_2]
            output[y_out * dim_output + x_out] = output_xy
    