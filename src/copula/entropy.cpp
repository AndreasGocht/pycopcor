#include <copula/entropy.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL copula_entropy_ARRAY_API
#include <numpy/arrayobject.h>

#include "exception.hpp"
#include "helper.hpp"

#pragma STDC FENV_ACCESS ON

PyObject *py_empirical_2d(PyObject *self, PyObject *args)
{
    PyObject *Fx_arg = NULL, *Fy_arg = NULL;
    int sample_points;
    double offset;
    npy_float64 result = 0;

    if (!PyArg_ParseTuple(args, "OOId", &Fx_arg, &Fy_arg, &sample_points, &offset))
    {
        return nullptr;
    }
    try
    {
        auto Fx = pyarray(Fx_arg);
        auto Fy = pyarray(Fy_arg);
        if (Fx.n != Fy.n)
        {
            throw std::runtime_error("Array lenght does not match");
        }
        error::floating_point::errors err;

        NPY_BEGIN_ALLOW_THREADS
        error::floating_point::clean();

        result = empirical_2d(Fx.data, Fy.data, Fx.n, sample_points, offset);

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

PyObject *py_empirical_2d_block(PyObject *self, PyObject *args)
{
    PyObject *Fx_arg = NULL, *Fy_arg = NULL;
    int sample_points;
    double offset;
    npy_float64 result = 0;

    if (!PyArg_ParseTuple(args, "OOId", &Fx_arg, &Fy_arg, &sample_points, &offset))
    {
        return nullptr;
    }
    try
    {
        auto Fx = pyarray(Fx_arg);
        auto Fy = pyarray(Fy_arg);
        if (Fx.n != Fy.n)
        {
            throw std::runtime_error("Array lenght does not match");
        }
        error::floating_point::errors err;

        NPY_BEGIN_ALLOW_THREADS
        error::floating_point::clean();

        result = empirical_2d_block(Fx.data, Fy.data, Fx.n, sample_points, offset);

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

PyObject *py_empiricial_3d(PyObject *self, PyObject *args)
{
    PyObject *Fx_arg = NULL, *Fy_arg = NULL, *Fz_arg = NULL;
    int sample_points;
    double offset;
    npy_float64 result = 0;

    if (!PyArg_ParseTuple(args, "OOOId", &Fx_arg, &Fy_arg, &Fz_arg, &sample_points, &offset))
    {
        return nullptr;
    }
    try
    {
        auto Fx = pyarray(Fx_arg);
        auto Fy = pyarray(Fy_arg);
        auto Fz = pyarray(Fz_arg);
        if ((Fx.n != Fy.n) or (Fy.n != Fz.n))
        {
            throw std::runtime_error("Array lenght does not match");
        }

        error::floating_point::errors err;

        NPY_BEGIN_ALLOW_THREADS
        error::floating_point::clean();

        result = empirical_3d(Fx.data, Fy.data, Fz.data, Fx.n, sample_points, offset);

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

PyObject *py_empiricial_3d_block(PyObject *self, PyObject *args)
{
    PyObject *Fx_arg = NULL, *Fy_arg = NULL, *Fz_arg = NULL;
    int sample_points;
    double offset;
    npy_float64 result = 0;

    if (!PyArg_ParseTuple(args, "OOOId", &Fx_arg, &Fy_arg, &Fz_arg, &sample_points, &offset))
    {
        return nullptr;
    }
    try
    {
        auto Fx = pyarray(Fx_arg);
        auto Fy = pyarray(Fy_arg);
        auto Fz = pyarray(Fz_arg);
        if ((Fx.n != Fy.n) or (Fy.n != Fz.n))
        {
            throw std::runtime_error("Array lenght does not match");
        }

        error::floating_point::errors err;

        NPY_BEGIN_ALLOW_THREADS
        error::floating_point::clean();

        result = empirical_3d_block(Fx.data, Fy.data, Fz.data, Fx.n, sample_points, offset);

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

static PyMethodDef copula_entropy_methods[] = {
    {"_empirical_2d", py_empirical_2d, METH_VARARGS, "Calculates the two dimensional Copula Entropy."},
    {"_empirical_2d_block", py_empirical_2d_block, METH_VARARGS, "Calculates the two dimensional Copula Entropy."},
    {"_empirical_3d", py_empiricial_3d, METH_VARARGS, "Calculates the three dimensional Copula Entropy."},
    {"_empirical_3d_block", py_empiricial_3d_block, METH_VARARGS, "Calculates the three dimensional Copula Entropy."},
    {NULL, NULL, 0, NULL} /*Sentinel*/
};

static struct PyModuleDef copula_entropy_module = {PyModuleDef_HEAD_INIT,
    "_entropy", /*name of module*/
    NULL,       /*module documentation,may be NULL*/
    -1,         /*size of per - interpreter state of the module, or
                   -1   if the module keeps state in global
                   variables.*/
    copula_entropy_methods};

extern "C"
{
    PyMODINIT_FUNC PyInit__entropy(void)
    {
        auto module = PyModule_Create(&copula_entropy_module);
        if (module == NULL)
            return NULL;
        import_array();
        return module;
    }
}
