#include <copula/wolff.h>
#include <functional>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL copula_wolff_ARRAY_API
#include <numpy/arrayobject.h>

#include "exception.hpp"
#include "helper.hpp"

#pragma STDC FENV_ACCESS ON

PyObject *parse_and_run(
    PyObject *self, PyObject *args, std::function<double(double *, double *, size_t, int)> wolff_fun)
{
    PyObject *X_arg = NULL, *Y_arg = NULL;
    int sample_points;
    npy_float64 result = 0;

    if (!PyArg_ParseTuple(args, "OOI", &X_arg, &Y_arg, &sample_points))
    {
        return nullptr;
    }
    try
    {
        auto X = pyarray(X_arg);
        auto Y = pyarray(Y_arg);
        if (X.n != Y.n)
        {
            throw std::runtime_error("Array lenght does not match");
        }
        error::floating_point::errors err;

        NPY_BEGIN_ALLOW_THREADS
        error::floating_point::clean();

        result = wolff_fun(X.data, Y.data, X.n, sample_points);

        err = error::floating_point::check();
        NPY_END_ALLOW_THREADS

        error::floating_point::report_to_python(err);
    }
    catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }

    return create_return_object(result);
}

PyObject *py_copula_wollf_gamma(PyObject *self, PyObject *args)
{
    return parse_and_run(self, args, copula_wollf_gamma);
}

PyObject *py_copula_wollf_sigma(PyObject *self, PyObject *args)
{
    return parse_and_run(self, args, copula_wollf_sigma);
}

PyObject *py_copula_wollf_spearman(PyObject *self, PyObject *args)
{
    return parse_and_run(self, args, copula_wollf_spearman);
}

static PyMethodDef copula_wolff_methods[] = {
    {"_gamma", py_copula_wollf_gamma, METH_VARARGS, "Calculates Wolffs gamma."},
    {"_sigma", py_copula_wollf_sigma, METH_VARARGS, "Calculates Wolffs sigma."},
    {"_spearman", py_copula_wollf_spearman, METH_VARARGS, "Calculates the Spreamans roh as noted by Wolff."},
    {NULL, NULL, 0, NULL} /*Sentinel*/
};

static struct PyModuleDef copula_wolff_module = {PyModuleDef_HEAD_INIT,
    "_wolff", /*name of module*/
    NULL,     /*module documentation,may be NULL*/
    -1,       /*size of per - interpreter state of the module, or
                 -1   if the module keeps state in global
                 variables.*/
    copula_wolff_methods};

extern "C"
{
    PyMODINIT_FUNC PyInit__wolff(void)
    {
        auto module = PyModule_Create(&copula_wolff_module);
        if (module == NULL)
            return NULL;
        import_array();
        return module;
    }
}
