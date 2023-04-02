#ifndef INCLUDE_COPCOR_COPULA_WOLFF_H_
#define INCLUDE_COPCOR_COPULA_WOLFF_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

    double copula_wollf_gamma(double *X, double *Y, size_t n, int sample_points);
    double copula_wollf_sigma(double *X, double *Y, size_t n, int sample_points);
    double copula_wollf_spearman(double *X, double *Y, size_t n, int sample_points);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_COPULA_WOLFF_H_ */
