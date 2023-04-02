
#include <stdlib.h>
#include <string.h>

#ifndef KBLOCK_SIZE
#define KBLOCK_SIZE 32
#endif

#ifndef LBLOCK_SIZE
#define LBLOCK_SIZE 256
#endif

__attribute__((target_clones("avx2", "default"))) void marginal_density(double *X, size_t n, double **ret_Fx)
{
    double *F_x = aligned_alloc(64, n * sizeof(double));

#pragma omp parallel for
    for (size_t k = 0; k < n; k++)
    {
        double X_k = X[k];
        double F_x_k = 0;

#pragma omp simd safelen(4) reduction(+ : F_x_k)
        for (size_t l = 0; l < n; l++)
        {
            F_x_k += (X[l] <= X_k);
        }

        F_x[k] = F_x_k / n;
    }

    *ret_Fx = F_x;
}

__attribute__((target_clones("avx2", "default"))) void marginal_density_block(double *X, size_t n, double **ret_Fx)
{
    double *F_x = aligned_alloc(64, n * sizeof(double));

#pragma omp parallel for
    for (size_t kk = 0; kk < n; kk += KBLOCK_SIZE)
    {
        size_t kk_end = ((kk + KBLOCK_SIZE) > n) ? n : (kk + KBLOCK_SIZE);
        for (size_t k = kk; k < kk_end; k++)
        {
            F_x[k] = 0;
        }

        for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
        {
            size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);
            for (size_t k = kk; k < kk_end; k++)
            {
                double X_k = X[k];
                double F_x_k = 0;

#pragma omp simd reduction(+ : F_x_k)
                for (size_t l = ll; l < ll_end; l++)
                {
                    F_x_k += (X[l] <= X_k);
                }
                F_x[k] += F_x_k;
            }
        }
        for (size_t k = kk; k < kk_end; k++)
        {
            F_x[k] = F_x[k] / n;
        }
    }

    *ret_Fx = F_x;
}
