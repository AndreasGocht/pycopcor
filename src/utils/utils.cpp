#include "exception.hpp"
#include "helper.hpp"

#include <iostream>
#include <limits>

#include <utils.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL util_ARRAY_API
#include <numpy/arrayobject.h>

#pragma STDC FENV_ACCESS ON

PyObject *py_bindings_and_min_distance(PyObject *self, PyObject *args)
{
    PyObject *X_arg = NULL;
    bool bindings = false;
    double min_distance = std::numeric_limits<double>::infinity();
    size_t *bindings_list = nullptr;
    size_t bindings_list_n = 0;

    if (!PyArg_ParseTuple(args, "O", &X_arg))
    {
        return nullptr;
    }
    try
    {
        auto X = pyarray(X_arg);
        error::floating_point::errors err;

        NPY_BEGIN_ALLOW_THREADS
        error::floating_point::clean();

        bindings = bindings_and_min_distance(X.data, X.n, &min_distance, &bindings_list);

        bindings_list_n = X.n;

        err = error::floating_point::check();
        NPY_END_ALLOW_THREADS

        error::floating_point::report_to_python(err);
    }
    catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }

    PyObject *res = PyTuple_New(3);
    PyTuple_SetItem(res, 0, PyBool_FromLong(bindings));
    PyTuple_SetItem(res, 1, create_return_object(min_distance));
    PyTuple_SetItem(res, 2, create_return_array_from_data(bindings_list, bindings_list_n));
    return res;
}

static PyMethodDef util_methods[] = {
    {"_bindings_and_min_distance", py_bindings_and_min_distance, METH_VARARGS, "Find bindings."},
    {NULL, NULL, 0, NULL} /*Sentinel*/
};

static struct PyModuleDef util_module = {PyModuleDef_HEAD_INIT,
    "_utils", /*name of module*/
    NULL,     /*module documentation,may be NULL*/
    -1,       /*size of per - interpreter state of the module, or
                 -1   if the module keeps state in global
                 variables.*/
    util_methods};

extern "C"
{
    PyMODINIT_FUNC PyInit__utils(void)
    {
        auto module = PyModule_Create(&util_module);
        if (module == NULL)
            return NULL;
        import_array();
        return module;
    }
}
