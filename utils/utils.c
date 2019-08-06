//
// @author andrii dobroshynskyi
//

#include "utils.h"

//
// flag for verbose output
//
int verbose = 0;

//
// memory management
//

// create a 2D array equivalent to a nxm matrix
//
float** create_array_2d(int n, int m)
{
    float* values = calloc(m * n, sizeof(float));
    float** rows = malloc(n * sizeof(float*));
    for(int i = 0; i < n; i++) {
        rows[i] = values + i * m;
    }
    return rows;
}

//
// free any 2D array
//
void destroy_array_2d(float** arr)
{
    free(*arr);
    free(arr);
}

//
// performance metrics calculation utilities
//
double average_accuracy(double* accuracy_vector, int n)
{
    double total = 0;
    for(int i=0; i < n; i++)
    {
        total += accuracy_vector[i];
    }
    return total/n;
}

double get_accuracy(int n, float* actual, float* prediction)
{
    int correct = 0;
    for(int i=0; i < n; i++)
    {
        if(actual[i] == prediction[i]) correct++;
    }
    return (correct * 1.0 / n * 1.0) * 1.0;
}

//
// utilities for array management
//
int contains_int(int* arr, int n, int val)
{
    for(int i=0; i < n; i++)
    {
        if(arr[i] == val) return 1;
    }
    return 0;
}
int contains_float(float* arr, int n, float val)
{
    for(int i=0; i < n; i++)
    {
        if(arr[i] == val) return 1;
    }
    return 0;
}

//
// utility to merge two arrays into one
//
float** combine_arrays(float** first, float** second, int n1, int n2, int cols)
{
    float** combined = (float**) malloc((n1 + n2) * sizeof(float) * cols);
    int row_index = 0;
    for(int i=0; i < n1; i++)
    {
        float* row = first[i];
        combined[row_index] = row;
        row_index++;
    }
    for(int j=0; j < n2; j++)
    {
        float* row = second[j];
        combined[row_index] = row;
        row_index++;
    }
    return combined;
}