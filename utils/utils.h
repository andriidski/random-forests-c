//
// @author andrii dobroshynskyi
//

#ifndef RANDOM_FOREST_C_UTILS_H
#define RANDOM_FOREST_C_UTILS_H

#include <stdlib.h>

//
// array utils
//
float** combine_arrays(float** first, float** second, int n1, int n2, int cols);

//
// accuracy utils
//
double average_accuracy(double* accuracy_vector, int n);
double get_accuracy(int n, float* actual, float* prediction);

//
// contains check utils
//
int contains_int(int* arr, int n, int val);
int contains_float(float* arr, int n, float val);

//
// memory management utils
//
float** create_array_2d(int n, int m);
void destroy_array_2d(float** arr);

extern int verbose;

#endif //RANDOM_FOREST_C_UTILS_H
