#ifndef INCLUDE_COPCOR_MARGINAL_DENSITY_H_
#define INCLUDE_COPCOR_MARGINAL_DENSITY_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif
    void marginal_density(double *X, size_t n, double **ret_Fx);
    void marginal_density_block(double *X, size_t n, double **ret_Fx);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_COPCOR_MARGINAL_DENSITY_H_ */
