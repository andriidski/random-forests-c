/*
@author andrii dobroshynski
*/

#ifndef data_h
#define data_h

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "utils.h"

/*
Struct for parsed data dimensions.
*/
struct dim
{
    size_t rows;
    size_t cols;
};

/*
Attempts to read a csv file at path given by 'file_name', and if successfull, records the
dimensions of the csv file, asserts that all rows have the same number of columns, and returns
the dimensions of the file in a struct of type dim.

If the numbers for rows and cols are given to the program as command line arguments, then calling
this function can be skipped, as we already would know how much memory to allocate to fit the csv
file of the given dimensions.
*/
struct dim parse_csv_dims(const char *file_name);

/*
Attempts to read a csv file at path given by 'file_name' and write the values one-by-one into 'data'.
If an argument for '--num_rows' was provided to the program and is less than the actual number of 
rows in the csv file, the function will stop reading at that row. This allows to read only the top
'num_rows' in the input data file if needed.
*/
void parse_csv(const char *file_name, double **data_p, const struct dim csv_dim);

/*
Pivots and transforms the data in 'data' array into a two-dimensional array of size 
'csv_dim.rows' * 'csv_dim.cols' pointed to by 'pivoted_data_p'.
*/
void pivot_data(double *data, const struct dim csv_dim, double ***pivoted_data_p);

#endif // data_h
