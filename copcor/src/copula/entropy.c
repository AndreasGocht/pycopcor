#include "entropy.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef ALIGMENT
#define ALIGMENT 64
#endif

__attribute__((target_clones("avx2", "default"))) double empirical_2d(
    double *F_x, double *F_y, size_t n, int sample_points, double offset)
{
    // O(n^2 + L^2 * n)

    double res = 0;

    double du = 1.0 / sample_points;
    double dv = 1.0 / sample_points;

    size_t mem_size_e = sample_points * n * sizeof(double);
    mem_size_e = mem_size_e % ALIGMENT == 0 ? mem_size_e : (mem_size_e / ALIGMENT + 1) * ALIGMENT;

    double *ex = aligned_alloc(ALIGMENT, mem_size_e);
    double *ey = aligned_alloc(ALIGMENT, mem_size_e);

#pragma omp parallel for shared(F_x, F_y, ex, ey) firstprivate(sample_points, n, du, dv) schedule(static)
    for (int i = 0; i < sample_points; i++)
    {
        double u = du * i;
        double v = dv * i;
#pragma omp simd aligned(ex, ey : ALIGMENT)
        for (size_t k = 0; k < n; k++)
        {
            ex[n * i + k] = exp((F_x[k] - u) * offset);
            ey[n * i + k] = exp((F_y[k] - v) * offset);
        }
    }

#pragma omp parallel for shared(ex, ey) firstprivate(sample_points, n, du, dv) reduction(+:res) collapse(2) schedule(static)
    for (int i = 0; i < sample_points; i++)
    {
        for (int j = 0; j < sample_points; j++)
        {

            // compute d/du d/dv C(u,v)
            // d/du d/dv (x <= u) * (y <= v)
            // d/du d/dv sigmoid(-(x-u)*o) sigmoid(-(y-v)*o)
            // d/(du)(d/(dv)1/((1 + exp(-(-(x - u) o))) (1 + exp(-(-(y - v) o)))))
            // = (o^2 e^(o (x - u) + o (y - v)))/((e^(o (x - u)) + 1)^2 (e^(o (y - v)) + 1)^2)
            // https://www.wolframalpha.com/input/?i=d%2Fdu+d%2Fdv+sigmoid%28-%28x-u%29*o%29+sigmoid%28-%28y-v%29*o%29

            double res_sample_point = 0;
#pragma omp simd aligned(ex, ey : ALIGMENT) reduction(+ : res_sample_point)
            for (size_t k = 0; k < n; k++)
            {
                double ex_curr = ex[n * i + k];
                double ey_curr = ey[n * j + k];
                double ex_1 = ex_curr + 1;
                double ey_1 = ey_curr + 1;

                res_sample_point += offset * offset * ex_curr * ey_curr / (ex_1 * ex_1 * ey_1 * ey_1);
            }
            double c_u = res_sample_point / n;

            double tmp_res = c_u * log(c_u) * du * dv;
            res += tmp_res;
        }
    }

    free(ex);
    free(ey);

    return res;
}

__attribute__((target_clones("avx2", "default"))) double empirical_2d_block(
    double *F_x, double *F_y, size_t n, int sample_points, double offset)
{
    // O(n^2 + L^2 * n)

    double res = 0;

    double du = 1.0 / sample_points;
    double dv = 1.0 / sample_points;

    size_t mem_size_e = sample_points * n * sizeof(double);
    mem_size_e = mem_size_e % ALIGMENT == 0 ? mem_size_e : (mem_size_e / ALIGMENT + 1) * ALIGMENT;

    double *ex = aligned_alloc(ALIGMENT, mem_size_e);
    double *ey = aligned_alloc(ALIGMENT, mem_size_e);

    size_t mem_size_block = sample_points * sizeof(double);
    mem_size_block = mem_size_block % ALIGMENT == 0 ? mem_size_block : (mem_size_block / ALIGMENT + 1) * ALIGMENT;

#pragma omp parallel for shared(F_x, F_y, ex, ey) firstprivate(sample_points, n, du, dv) schedule(static)
    for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
    {
        size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);
        for (int i = 0; i < sample_points; i++)
        {
            double u = du * i;
            double v = dv * i;
#pragma omp simd aligned(ex, ey : ALIGMENT)
            for (size_t l = ll; l < ll_end; l++)
            {
                ex[n * i + l] = exp((F_x[l] - u) * offset);
                ey[n * i + l] = exp((F_y[l] - v) * offset);
            }
        }
    }

    double offset_2 = offset * offset;

#pragma omp parallel for shared(ex, ey) firstprivate(sample_points, n, du, dv) reduction(+ : res) schedule(static)
    for (int i = 0; i < sample_points; i++)
    {
        size_t ni = n * i;

        // double *j_samples = (double *) calloc(sample_points, sizeof(double));
        double *j_samples = (double *) aligned_alloc(ALIGMENT, mem_size_block);
        memset(j_samples, 0, mem_size_block);

        for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
        {
            size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);

            for (int j = 0; j < sample_points; j++)
            {
                size_t nj = n * j;

                double res_sample_point = 0;
#pragma omp simd aligned(ex, ey : ALIGMENT) reduction(+ : res_sample_point)
                for (size_t l = ll; l < ll_end; l++)
                {
                    // compute d/du d/dv C(u,v)
                    // d/du d/dv (x <= u) * (y <= v)
                    // d/du d/dv sigmoid(-(x-u)*o) sigmoid(-(y-v)*o)
                    // d/(du)(d/(dv)1/((1 + exp(-(-(x - u) o))) (1 + exp(-(-(y - v) o)))))
                    // = (o^2 e^(o (x - u) + o (y - v)))/((e^(o (x - u)) + 1)^2 (e^(o (y - v)) + 1)^2)
                    // https://www.wolframalpha.com/input/?i=d%2Fdu+d%2Fdv+sigmoid%28-%28x-u%29*o%29+sigmoid%28-%28y-v%29*o%29

                    double ex_curr = ex[ni + l];
                    double ey_curr = ey[nj + l];
                    double ex_1 = ex_curr + 1;
                    double ey_1 = ey_curr + 1;

                    res_sample_point += offset_2 * ex_curr * ey_curr / (ex_1 * ex_1 * ey_1 * ey_1);
                }
                j_samples[j] += res_sample_point;
            }
        }

#pragma omp simd aligned(j_samples : ALIGMENT) reduction(+ : res)
        for (int j = 0; j < sample_points; j++)
        {
            double c_u = j_samples[j] / n;
            double tmp_res = c_u * log(c_u) * du * dv;
            res += tmp_res;
        }
        free(j_samples);
    }

    free(ex);
    free(ey);

    return res;
}

__attribute__((target_clones("avx2", "default"))) double empirical_3d(
    double *F_x, double *F_y, double *F_z, size_t n, int sample_points, double offset)
{
    // O(n^2 + L^3 * n) = O(L^3 * n) ... Worst case
    double res = 0;
    double du = 1.0 / sample_points;
    double dv = 1.0 / sample_points;
    double dw = 1.0 / sample_points;

    size_t mem_size_e = sample_points * n * sizeof(double);
    mem_size_e = mem_size_e % ALIGMENT == 0 ? mem_size_e : (mem_size_e / ALIGMENT + 1) * ALIGMENT;

    double *ex = aligned_alloc(ALIGMENT, mem_size_e);
    double *ey = aligned_alloc(ALIGMENT, mem_size_e);
    double *ez = aligned_alloc(ALIGMENT, mem_size_e);

#pragma omp parallel for shared(F_x, F_y, F_z, ex, ey, ez) firstprivate(sample_points, n, du, dv, dw) schedule(static)
    for (int i = 0; i < sample_points; i++)
    {

#pragma omp simd aligned(ex, ey, ez : ALIGMENT)
        for (size_t l = 0; l < n; l++)
        {
            double u = du * i;
            double v = dv * i;
            double w = dw * i;

            ex[n * i + l] = exp((F_x[l] - u) * offset);
            ey[n * i + l] = exp((F_y[l] - v) * offset);
            ez[n * i + l] = exp((F_z[l] - w) * offset);
        }
    }

#pragma omp parallel for shared(ex, ey, ez) firstprivate(sample_points, n, du, dv, dw) reduction(+:res) collapse(3) schedule(static)
    for (int i = 0; i < sample_points; i++)
    {
        for (int j = 0; j < sample_points; j++)
        {
            for (int k = 0; k < sample_points; k++)
            {

                // compute d/du d/dv C(u,v)
                // d/du d/dv d/dw (x <= u) * (y <= v) * (z <= w)
                // d/du d/dv d/dw sigmoid(-(x-u)*o) sigmoid(-(y-v)*o)  sigmoid(-(z-w)*o)
                //
                // d/(du)(d/(dv)d/(dw)
                // 1/((1 + exp(-(-(x - u) o))) (1 + exp(-(-(y - v) o))) (1 + exp(-(-(z - w) o)))))
                // = (o^3 e^(o (x - u) + o (y - v) + o (z - w)))
                //   /((e^(o (x - u)) + 1)^2 (e^(o (y - v)) + 1)^2 (e^(o (z - w)) + 1)^2)
                //
                // https://www.wolframalpha.com/input/?i=d%2Fdu+d%2Fdv+d%2Fdw+sigmoid%28-%28x-u%29*o%29+sigmoid%28-%28y-v%29*o%29++sigmoid%28-%28z-w%29*o%29

                double res_sample_point = 0;
#pragma omp simd aligned(ex, ey, ez : ALIGMENT) reduction(+ : res_sample_point)
                for (size_t l = 0; l < n; l++)
                {
                    double ex_curr = ex[n * i + l];
                    double ey_curr = ey[n * j + l];
                    double ez_curr = ez[n * k + l];
                    double ex_1 = ex_curr + 1;
                    double ey_1 = ey_curr + 1;
                    double ez_1 = ez_curr + 1;

                    res_sample_point += offset * offset * offset * ex_curr * ey_curr * ez_curr /
                                        (ex_1 * ex_1 * ey_1 * ey_1 * ez_1 * ez_1);
                }

                double c_u = res_sample_point / n;
                double tmp_res = c_u * log(c_u) * du * dv * dw;
                res += tmp_res;
            }
        }
    }

    free(ex);
    free(ey);
    free(ez);

    return res;
}

__attribute__((target_clones("avx2", "default"))) double empirical_3d_block(
    double *F_x, double *F_y, double *F_z, size_t n, int sample_points, double offset)
{
    double res = 0;
    double du = 1.0 / sample_points;
    double dv = 1.0 / sample_points;
    double dw = 1.0 / sample_points;

    size_t mem_size_e = sample_points * n * sizeof(double);
    mem_size_e = mem_size_e % ALIGMENT == 0 ? mem_size_e : (mem_size_e / ALIGMENT + 1) * ALIGMENT;

    double *ex = aligned_alloc(ALIGMENT, mem_size_e);
    double *ey = aligned_alloc(ALIGMENT, mem_size_e);
    double *ez = aligned_alloc(ALIGMENT, mem_size_e);

    size_t mem_size_block = sample_points * sizeof(double);
    mem_size_block = mem_size_block % ALIGMENT == 0 ? mem_size_block : (mem_size_block / ALIGMENT + 1) * ALIGMENT;

#pragma omp parallel for shared(F_x, F_y, F_z, ex, ey, ez) firstprivate(sample_points, n, du, dv, dw) schedule(static)
    for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
    {
        size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);

        for (int i = 0; i < sample_points; i++)
        {
            double u = du * i;
            double v = dv * i;
            double w = dw * i;

#pragma omp simd aligned(ex, ey, ez : ALIGMENT)
            for (size_t l = ll; l < ll_end; l++)
            {
                ex[n * i + l] = exp((F_x[l] - u) * offset);
                ey[n * i + l] = exp((F_y[l] - v) * offset);
                ez[n * i + l] = exp((F_z[l] - w) * offset);
            }
        }
    }

    double offset_3 = offset * offset * offset;

#pragma omp parallel for shared(ex, ey, ez) firstprivate(sample_points, n, du, dv, dw) reduction(+:res) collapse(2) schedule(static)
    for (int i = 0; i < sample_points; i++)
    {
        for (int j = 0; j < sample_points; j++)
        {
            size_t ni = n * i;
            size_t nj = n * j;

            // double *k_samples = (double *) calloc(sample_points, sizeof(double));
            double *k_samples = (double *) aligned_alloc(ALIGMENT, mem_size_block);
            memset(k_samples, 0, mem_size_block);

            for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
            {
                size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);

                for (int k = 0; k < sample_points; k++)
                {

                    // compute d/du d/dv C(u,v)
                    // d/du d/dv d/dw (x <= u) * (y <= v) * (z <= w)
                    // d/du d/dv d/dw sigmoid(-(x-u)*o) sigmoid(-(y-v)*o)  sigmoid(-(z-w)*o)
                    //
                    // d/(du)(d/(dv)d/(dw)
                    // 1/((1 + exp(-(-(x - u) o))) (1 + exp(-(-(y - v) o))) (1 + exp(-(-(z - w) o)))))
                    // = (o^3 e^(o (x - u) + o (y - v) + o (z - w)))
                    //   /((e^(o (x - u)) + 1)^2 (e^(o (y - v)) + 1)^2 (e^(o (z - w)) + 1)^2)
                    //
                    // https://www.wolframalpha.com/input/?i=d%2Fdu+d%2Fdv+d%2Fdw+sigmoid%28-%28x-u%29*o%29+sigmoid%28-%28y-v%29*o%29++sigmoid%28-%28z-w%29*o%29

                    size_t nk = n * k;
                    double res_sample_point = 0;

#pragma omp simd aligned(ex, ey, ez : ALIGMENT) reduction(+ : res_sample_point)
                    for (size_t l = ll; l < ll_end; l++)
                    {
                        double ex_curr = ex[ni + l];
                        double ey_curr = ey[nj + l];
                        double ez_curr = ez[nk + l];
                        double ex_1 = ex_curr + 1;
                        double ey_1 = ey_curr + 1;
                        double ez_1 = ez_curr + 1;

                        res_sample_point +=
                            offset_3 * ex_curr * ey_curr * ez_curr / (ex_1 * ex_1 * ey_1 * ey_1 * ez_1 * ez_1);
                    }
                    k_samples[k] += res_sample_point;
                }
            }

#pragma omp simd aligned(k_samples : ALIGMENT) reduction(+ : res)
            for (int k = 0; k < sample_points; k++)
            {
                double c_u = k_samples[k] / n;
                double tmp_res = c_u * log(c_u) * du * dv * dw;
                res += tmp_res;
            }
            free(k_samples);
        }
    }

    free(ex);
    free(ey);
    free(ez);

    return res;
}