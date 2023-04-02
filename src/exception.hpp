/*
 * exception.hpp
 *
 *  Created on: 28.05.2019
 *      Author: gocht
 */

#ifndef INCLUDE_CMI_EXCEPTION_HPP_
#define INCLUDE_CMI_EXCEPTION_HPP_

#include <Python.h>
#include <cfenv>
#include <stdexcept>

namespace error::floating_point
{
struct errors
{
    bool inexact = false;
    bool invalid = false;
    bool overflow = false;
    bool underflow = false;
    bool divbyzero = false;
};

inline void clean()
{
#pragma omp parallel
    {
        // clean floating point exceptions
        std::feclearexcept(FE_ALL_EXCEPT);
    }
}

inline errors check()
{
    errors err;
#pragma omp parallel shared(err)
    {
#pragma omp critical
        {
            if (std::fetestexcept(FE_DIVBYZERO))
            {
                err.divbyzero = true;
            }
            if (std::fetestexcept(FE_INEXACT))
            {
                err.inexact = true;
            }
            if (std::fetestexcept(FE_INVALID))
            {
                err.invalid = true;
            }
            if (std::fetestexcept(FE_OVERFLOW))
            {
                err.overflow = true;
            }
            if (std::fetestexcept(FE_UNDERFLOW))
            {
                err.underflow = true;
            }
        }

        // clean floating point exceptions
        std::feclearexcept(FE_ALL_EXCEPT);
    }
    return err;
}

inline void report_to_python(errors err)
{
    if (err.divbyzero)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning, "Division by Zero encountered", 1);
    }
    if (err.invalid)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning, "Invalid result encountered", 1);
    }
    // its always inexact ... floating point arithmetics
    // if (err.inexact)
    // {
    //     PyErr_WarnEx(PyExc_RuntimeWarning, "Inexact result encountered", 1);
    // }
    if (err.overflow)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning, "Overflow encountered", 1);
    }
    if (err.overflow)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning, "Underflow encountered", 1);
    }
}
} // namespace error::floating_point

#endif /* INCLUDE_CMI_EXCEPTION_HPP_ */
