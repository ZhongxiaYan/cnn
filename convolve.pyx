import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_forward(np.ndarray[np.float_t, ndim=3] input, np.ndarray[np.float_t, ndim=3] W, int dim_input, int dim_W, int padding, int stride, np.ndarray[np.float_t, ndim=3] output):
    '''
    input: <batch_size> * <input_depth> * <dim_input ^ 2>
    W: <output_depth> * <input_depth> * <dim_W ^ 2>
    padding: padding on left, right, top, and bottom
    output: <batch_size> * <output_depth> * <dim_output ^ 2>
    '''
    output.fill(0)
    cdef int batch_size = input.shape[0]
    cdef int output_depth = output.shape[1]
    cdef int input_depth = input.shape[1]
    cdef int dim_output = (dim_input - dim_W + 2 * padding) // stride + 1
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef int x, y, z # respectively, indices in batch_size, output_depth, input_depth
    # memory views for efficient accessing
    cdef np.float_t [:, :, :] input_buffer = input
    cdef np.float_t [:, :, :] output_buffer = output
    cdef np.float_t [:, :, :] W_buffer = W
    cdef np.float_t [:, :] input_x, output_x, W_y
    cdef np.float_t [:] input_xz, output_xy, W_yz
    cdef np.float_t output_xymn
    for x in range(batch_size):
        input_x = input_buffer[x]
        output_x = output_buffer[x]
        for y in range(output_depth):
            W_y = W_buffer[y]
            output_xy = output_x[y]
            for z in range(input_depth):
                W_yz = W_y[z]
                input_xz = input_x[z]
                for m in range(dim_output):
                    i = m * stride - padding
                    for n in range(dim_output):
                        j = n * stride - padding
                        output_xymn = 0
                        for a in range(dim_W):
                            ia = (i + a)
                            if ia < 0 or ia >= dim_input:
                                continue
                            for b in range(dim_W):
                                jb = (j + b)
                                if jb < 0 or jb >= dim_input:
                                    continue
                                output_xymn += input_xz[ia * dim_input + jb] * W_yz[a * dim_W + b]
                        output_xy[m * dim_output + n] += output_xymn

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_backward_W(np.ndarray[np.float_t, ndim=3] dJ_dout, np.ndarray[np.float_t, ndim=3] input, int dim_input, int dim_W, int padding, int stride, np.ndarray[np.float_t, ndim=3] dJ_dW):
    '''
    dJ_dout: <batch_size> * <output_depth> * <dim_input ^ 2>
    input: <batch_size> * <input_depth> * <dim_input ^ 2>
    dJ_dW: <output_depth> * <input_depth> * <dim_W ^ 2>
    '''
    dJ_dW.fill(0)
    cdef int batch_size = input.shape[0]
    cdef int output_depth = dJ_dout.shape[1]
    cdef int input_depth = input.shape[1]
    cdef int dim_output = (dim_input - dim_W + 2 * padding) // stride + 1
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef int x, y, z # respectively, indices in batch_size, output_depth, input_depth
    # memory views for efficient accessing
    cdef np.float_t [:, :, :] dJ_dout_buffer = dJ_dout
    cdef np.float_t [:, :, :] input_buffer = input
    cdef np.float_t [:, :, :] dJ_dW_buffer = dJ_dW
    cdef np.float_t [:, :] dJ_dout_x, input_x, dJ_dW_y
    cdef np.float_t [:] input_xz, dJ_dout_xy, dJ_dW_yz
    cdef np.float_t dJ_dW_yzab

    for x in range(batch_size):
        dJ_dout_x = dJ_dout_buffer[x]
        input_x = input_buffer[x]
        for y in range(output_depth):
            dJ_dW_y = dJ_dW_buffer[y]
            dJ_dout_xy = dJ_dout_x[y]
            for z in range(input_depth):
                dJ_dW_yz = dJ_dW_y[z]
                input_xz = input_x[z]
                for a in range(dim_W):
                    for b in range(dim_W):
                        dJ_dW_yzab = 0
                        for m in range(dim_output):
                            i = m * stride - padding
                            ia = (i + a)
                            if ia < 0 or ia >= dim_input:
                                continue
                            for n in range(dim_output):
                                j = n * stride - padding
                                jb = (j + b)
                                if jb < 0 or jb >= dim_input:
                                    continue
                                dJ_dW_yzab += input_xz[ia * dim_input + jb] * dJ_dout_xy[m * dim_output + n]
                        dJ_dW_yz[a * dim_W + b] += dJ_dW_yzab

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_backward_input(np.ndarray[np.float_t, ndim=3] dJ_dout, np.ndarray[np.float_t, ndim=3] W, int dim_input, int dim_W, int padding, int stride, np.ndarray[np.float_t, ndim=3] dJ_din):
    dJ_din.fill(0)
    cdef int batch_size = dJ_din.shape[0]
    cdef int output_depth = dJ_dout.shape[1]
    cdef int input_depth = dJ_din.shape[1]
    cdef int dim_output = (dim_input - dim_W + 2 * padding) // stride + 1
    cdef int m, n # indices in output
    cdef int i, j # indices in input
    cdef int a, b # indices in W
    cdef int ia, jb # temp variables
    cdef int ip, jp # temp variables
    cdef int m_iter, n_iter, a_iter, b_iter # iteration variables
    cdef int x, y, z # respectively, indices in batch_size, output_depth, input_depth
    # memory views for efficient accessing
    cdef np.float_t [:, :, :] dJ_dout_buffer = dJ_dout
    cdef np.float_t [:, :, :] dJ_din_buffer = dJ_din
    cdef np.float_t [:, :, :] W_buffer = W
    cdef np.float_t [:, :] dJ_dout_x, dJ_din_x, W_y
    cdef np.float_t [:] dJ_dout_xy, W_yz, dJ_din_xz
    cdef np.float_t dJ_din_xzij

    for x in range(batch_size):
        dJ_dout_x = dJ_dout_buffer[x]
        dJ_din_x = dJ_din_buffer[x]
        for y in range(output_depth):
            W_y = W_buffer[y]
            dJ_dout_xy = dJ_dout_x[y]
            for z in range(input_depth):
                W_yz = W_y[z]
                dJ_din_xz = dJ_din_x[z]
                for i in range(dim_input):
                    ip = i + padding
                    m = ip // stride
                    a = ip % stride
                    if m >= dim_output:
                        a += stride * (m - dim_output + 1)
                        m = dim_output - 1
                    for j in range(dim_input):
                        jp = j + padding
                        n = jp // stride
                        b = jp % stride
                        if n >= dim_output:
                            b += stride * (n - dim_output + 1)
                            n = dim_output - 1
                        dJ_din_xzij = 0
                        m_iter = m
                        a_iter = a
                        while m_iter >= 0 and a_iter < dim_W:
                            n_iter = n
                            b_iter = b
                            while n_iter >= 0 and b_iter < dim_W:
                                dJ_din_xzij += dJ_dout_xy[m_iter * dim_output + n_iter] * W_yz[a_iter * dim_W + b_iter]
                                n_iter -= 1
                                b_iter += stride
                            m_iter -= 1
                            a_iter += stride
                        dJ_din_xz[i * dim_input + j] += dJ_din_xzij
