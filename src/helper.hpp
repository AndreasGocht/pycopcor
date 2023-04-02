#ifndef INCLUDE_HELPER_HPP_
#define INCLUDE_HELPER_HPP_

#include <Python.h>
#include <stdexcept>
#include <type_traits>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

template <class... T> constexpr bool always_false = false;

struct pyarray
{
    pyarray(PyObject *arg)
    {
        array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
        if (array == NULL)
        {
            throw std::runtime_error("Error parsing Numpy Array");
        }
        if (PyArray_NDIM(array) != 1)
        {
            Py_XDECREF(array);
            throw std::runtime_error("Numpy array has to much dimension");
        }
        n = PyArray_DIMS(array)[0];
        data = static_cast<npy_float64 *>(PyArray_GETPTR1(array, 0));
    }

    ~pyarray()
    {
        Py_XDECREF(array);
    }

    PyArrayObject *array = nullptr;
    size_t n = 0;
    npy_float64 *data = nullptr;
};

template <typename T> PyObject *create_return_object(T value)
{
    PyObject *result = nullptr;
    if constexpr (std::is_same_v<T, double>)
    {
        result = PyFloat_FromDouble(value);
    }
    else
    {
        static_assert(always_false<T>);
    }
    return result;
}

template <typename T> PyObject *create_return_array_from_data(T *data, size_t n)
{
    npy_intp dims[] = {n};
    int ndims = 1;
    PyArrayObject *py_arr = nullptr;
    if constexpr (std::is_same_v<T, npy_float64>)
    {
        py_arr = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(ndims, dims, NPY_FLOAT64, data));
    }
    else if constexpr (std::is_same_v<T, npy_uintp>)
    {
        py_arr = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(ndims, dims, NPY_UINTP, data));
    }
    else
    {
        static_assert(always_false<T>, "T not supported");
    }

    return reinterpret_cast<PyObject *>(py_arr);
}

#endif