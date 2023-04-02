#ifndef INCLUDE_COPCOR_COPULA_ENTROPY_H_
#define INCLUDE_COPCOR_COPULA_ENTROPY_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

    double empirical_2d(double *X, double *Y, size_t n, int sample_points, double offset);
    double empirical_2d_block(double *X, double *Y, size_t n, int sample_points, double offset);
    double empirical_3d(double *F_x, double *F_y, double *F_z, size_t n, int sample_points, double offset);
    double empirical_3d_block(double *F_x, double *F_y, double *F_z, size_t n, int sample_points, double offset);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_COPULA_ENTROPY_H_ */
