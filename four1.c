#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define tau 6.2831855f

#define SWAP(a,b)  do  {\
    tempr = (a);        \
    (a) = (b);          \
    (b) = tempr;        \
} while (0)

//  four1 FFT from Numerical Recipes in C, p. 507 - 508.
void four1(double data[], size_t nn, int isign) {
    size_t n, mmax, m, j, istep, i;
    double wtemp, wr, wpr, wpi, wi, theta;
    double tempr, tempi;
    n = nn << 1;
    j = 1;
    for (i = 1; i < n; i += 2) {
        if (j > i) {
            SWAP(data[j], data[i]);
            SWAP(data[j+1], data[i+1]);
        }
        m = nn;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    mmax = 2;
    while (n > mmax) {
        istep = mmax << 1;
        theta = isign * (6.28318530717959 / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= n; i += istep) {
                j = i + mmax;
                tempr = wr * data[j] - wi * data[j+1];
                tempi = wr * data[j+1] + wi * data[j];
                data[j] = data[i] - tempr;
                data[j+1] = data[i+1] - tempi;
                data[i] += tempr;
                data[i+1] += tempi;
            }
            wr = (wtemp = wr) * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

static PyObject *pyfour1(PyObject *self, PyObject *args) {
    PyObject *lst;
    double *data;
    size_t nn, i, size;
    int isign;

    if (!PyArg_ParseTuple(args, "OKi", &lst, &nn, &isign))
        return NULL;
    
    size = PyList_Size(lst);
    data = malloc(size * sizeof(double));
    
    for (i = 0; i < size; ++i)
        data[i] = PyFloat_AsDouble(PyNumber_Float(PyList_GetItem(lst, i)));
    
    four1(data - 1, nn, isign);
    
    for (i = 0; i < size; ++i)
        PyList_SetItem(lst, i, PyFloat_FromDouble(data[i]));
    
    free(data);
    
    return Py_NewRef(lst);
}

static PyMethodDef Four1Methods[] = {
    {
        "four1",
        pyfour1,
        METH_VARARGS,
        "four1 FFT from Numerical Recipes in C, p. 507 - 508.\n"
        "Function is changed to return data for call chaining.\n"
        "Note that 1-based indexing is already factored into this algorithm,\n"
        "so no adjustments need to be made.\n"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef four1module = {
    PyModuleDef_HEAD_INIT,
    "four1",
    NULL,
    -1,
    Four1Methods
};

PyMODINIT_FUNC PyInit_four1(void){
    return PyModule_Create(&four1module);
}
