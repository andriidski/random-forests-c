/*
@author andrii dobroshynski
*/

#ifndef utils_h
#define utils_h

#include <stdlib.h>
#include "data.h"

/*
Struct to hold information about a current model training run.
*/
struct ModelContext
{
    const size_t testingFoldIdx;
    const size_t rowsPerFold;
};

typedef struct ModelContext ModelContext;

/*
The debug log level that can be adjusted via an argument.
*/
int log_level;

/*
Given a pointer to a buffer array of integers returns whether or not a given integer 'n'
is present in the array.
*/
int contains_int(int *arr, size_t n, int val);

/*
Given two two-dimensional like arrays 'first' and 'second', merges them into one and returns a
new pointer.
*/
double **combine_arrays(double **first, double **second, size_t n1, size_t n2, size_t cols);

/*
Given a row number and a model context returns whether or not the particular
row belongs to a fold that is designated as the evaluation / testing fold.
*/
int is_row_part_of_testing_fold(int row, const ModelContext *ctx);

/*
Sets the 'log_level' to the 'selected_log_level'.
*/
void set_log_level(int selected_log_level);

/*
Returns whatever the current 'log_level' is.
*/
int get_log_level();

/*
Allocates memory for a two-dimensional like array of size 'rows' * 'cols'.
*/
double **_2d_malloc(const size_t rows, const size_t cols);

/*
Allocates memory for a two-dimensional like array of size 'rows' * 'cols' with all elements initialized
to zero.
*/
double **_2d_calloc(const size_t rows, const size_t cols);

/*
Computes a checksum of a one-dimensional like array. Used to verify consistency of data.
*/
double _1d_checksum(double *data, size_t size);

/*
Computes a checksum of a two-dimensional like array. Used to verify consistency of data.
*/
double _2d_checksum(double **data, size_t rows, size_t cols);

#endif // utils_h
