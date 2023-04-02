#ifndef INCLUDE_COPCOR_UTIL_H_
#define INCLUDE_COPCOR_UTIL_H_

#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Check the given array for bindings and returns the mininal distance between points as
     * well as the amount of bindings per element.
     *
     * TODO:
     *  * Python interface and Test
     *  * Check for IEEE floating point, e.g. if min_distance is smaller, than what the largest float
     * can represent
     *
     * @param a [in] array with the data to check for bindings
     * @param n [in] length of the array to check for bindings
     * @param min_distance [out] minimal distance between two values.
     * @param bindings [out] count of values in @param a that have bindings.
     * @return bool true if bindings are found, else false
     */
    bool bindings_and_min_distance(double *a, size_t n, double *min_distance, size_t **bindings);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_UTIL_H_ */
