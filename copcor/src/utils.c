#include "utils.h"

#include <math.h>
#include <stdio.h>

bool bindings_and_min_distance(double *a, size_t n, double *min_distance, size_t **bindings)
{

    size_t block_size = 1024;
    double tmp_min_distance = INFINITY;
    size_t *local_bindings = (size_t *) calloc(n, sizeof(size_t));
    bool bindings_found = false;

#pragma omp parallel for reduction(min : tmp_min_distance) reduction(|| : bindings_found)
    for (size_t jj = 0; jj < n; jj += block_size)
    {
        // block about j (jj) to improve cach reusage

        size_t jj_end = jj + block_size;
        if (jj_end > n)
        {
            jj_end = n; // make sure the last block fits
        }

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = jj; j < jj_end; j++)
            {
                if (i == j)
                {
                    continue;
                }
                if (a[i] == a[j])
                {
                    bindings_found = true;
                    local_bindings[j]++; // use j to avoid collision, as we parralise about jj
                }
                else
                {
                    double d = fabs(a[i] - a[j]);
                    tmp_min_distance = fmin(tmp_min_distance, d);
                }
            }
        }
    }
    *min_distance = tmp_min_distance;
    *bindings = local_bindings;
    return bindings_found;
}
