#include "entropy.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef ALIGMENT
#define ALIGMENT 64
#endif

__attribute__((target_clones("avx2", "default"))) double copula_wollf_gamma(
    double *F_x, double *F_y, size_t n, int sample_points)
{
    if (n < (size_t) sample_points * 10)
    {
        fprintf(stderr, "Warning: copula_wollf_gamma might not work correctly if n >> L is not given\n");
    }

    double res = 0;

    double du = 1.0 / sample_points;
    double dv = 1.0 / sample_points;
    size_t mem_size = sample_points * sizeof(double);
    mem_size = mem_size % ALIGMENT == 0 ? mem_size : (mem_size / ALIGMENT + 1) * ALIGMENT;

#pragma omp parallel for shared(F_x, F_y) firstprivate(sample_points, n, du, dv) reduction(+ : res)
    for (int i = 1; i <= sample_points; i++)
    {
        double u = du * i;
        // double *j_samples = (double *) calloc(sample_points, sizeof(double));
        double *j_samples = (double *) aligned_alloc(ALIGMENT, mem_size);
        memset(j_samples, 0, mem_size);

        for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
        {
            size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);

            for (int j = 1; j <= sample_points; j++)
            {
                double res_sample_point = 0;

                double v = dv * j;
                // compute C(u,v)
#pragma omp simd aligned(j_samples : ALIGMENT) reduction(+ : res_sample_point)
                for (size_t l = ll; l < ll_end; l++)
                {
                    // compute F_x
                    res_sample_point += ((F_x[l] <= u) * (F_y[l] <= v));
                }
                j_samples[j] += res_sample_point;
            }
        }
#pragma omp simd aligned(j_samples : ALIGMENT) reduction(+ : res)
        for (int j = 0; j < sample_points; j++)
        {
            double v = dv * j;
            double res_sample_points = j_samples[j] / n;
            res_sample_points -= (u * v);
            res += res_sample_points * res_sample_points;
        }
        free(j_samples);
    }

    res = sqrt(90.0 / ((double) sample_points * (double) sample_points) * res);

    return res;
}

__attribute__((target_clones("avx2", "default"))) double copula_wollf_sigma(
    double *F_x, double *F_y, size_t n, int sample_points)
{
    if (n < (size_t) sample_points * 10)
    {
        fprintf(stderr, "Warning: copula_wollf_sigma might not work correctly if n >> L is not given\n");
    }

    double res = 0;
    double du = 1.0 / sample_points;
    double dv = 1.0 / sample_points;
    size_t mem_size = sample_points * sizeof(double);
    mem_size = mem_size % ALIGMENT == 0 ? mem_size : (mem_size / ALIGMENT + 1) * ALIGMENT;

#pragma omp parallel for shared(F_x, F_y) firstprivate(sample_points, n, du, dv) reduction(+ : res)
    for (int i = 1; i <= sample_points; i++)
    {
        double u = du * (i);
        // double *j_samples = (double *) calloc(sample_points, sizeof(double));
        double *j_samples = (double *) aligned_alloc(ALIGMENT, mem_size);
        memset(j_samples, 0, mem_size);

        for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
        {
            size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);

            for (int j = 1; j <= sample_points; j++)
            {
                double res_sample_point = 0;
                double v = dv * (j);
#pragma omp simd aligned(j_samples : ALIGMENT) reduction(+ : res_sample_point)
                for (size_t l = ll; l < ll_end; l++)
                {
                    res_sample_point += ((F_x[l] <= u) * (F_y[l] <= v));
                }
                j_samples[j] += res_sample_point;
            }
        }
#pragma omp simd aligned(j_samples : ALIGMENT) reduction(+ : res)
        for (int j = 0; j < sample_points; j++)
        {
            double v = dv * j;
            double res_sample_points = j_samples[j] / n;
            res_sample_points -= (u * v);
            res += fabs(res_sample_points);
        }
        free(j_samples);
    }
    res = 12.0 / ((double) sample_points * (double) sample_points) * res;
    return res;
}

__attribute__((target_clones("avx2", "default"))) double copula_wollf_spearman(
    double *F_x, double *F_y, size_t n, int sample_points)
{
    if (n < (size_t) sample_points * 10)
    {
        fprintf(stderr, "Warning: copula_wollf_spearman might not work correctly if n >> L is not given\n");
    }

    double res = 0;

    double du = 1.0 / sample_points;
    double dv = 1.0 / sample_points;
    size_t mem_size = sample_points * sizeof(double);
    mem_size = mem_size % ALIGMENT == 0 ? mem_size : (mem_size / ALIGMENT + 1) * ALIGMENT;

#pragma omp parallel for shared(F_x, F_y) firstprivate(sample_points, n, du, dv) reduction(+ : res)
    for (int i = 1; i <= sample_points; i++)
    {
        double u = du * (i);
        // double *j_samples = (double *) calloc(sample_points, sizeof(double));
        double *j_samples = (double *) aligned_alloc(ALIGMENT, mem_size);
        memset(j_samples, 0, mem_size);

        for (size_t ll = 0; ll < n; ll += BLOCK_SIZE)
        {
            size_t ll_end = ((ll + BLOCK_SIZE) > n) ? n : (ll + BLOCK_SIZE);

            for (int j = 1; j <= sample_points; j++)
            {
                double res_sample_point = 0;
                double v = dv * (j);
#pragma omp simd aligned(j_samples : ALIGMENT) reduction(+ : res_sample_point)
                for (size_t l = ll; l < ll_end; l++)
                {
                    res_sample_point += ((F_x[l] <= u) * (F_y[l] <= v));
                }
                j_samples[j] += res_sample_point;
            }
        }
#pragma omp simd aligned(j_samples : ALIGMENT) reduction(+ : res)
        for (int j = 0; j < sample_points; j++)
        {
            double v = dv * j;
            double res_sample_point = j_samples[j] / n;
            res_sample_point -= (u * v);
            res += res_sample_point;
        }
        free(j_samples);
    }

    res = 12.0 / ((double) sample_points * (double) sample_points) * res;

    return res;
}
