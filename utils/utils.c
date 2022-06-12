/*
@author andrii dobroshynski
*/

#include "utils.h"

int get_log_level()
{
    return log_level;
}

void set_log_level(int selected_log_level)
{
    if (selected_log_level < 0 || selected_log_level > 3)
    {
        printf("Error: log_level must be in range [0, 3] got: %d\n", selected_log_level);
        exit(1);
    }
    log_level = selected_log_level;
}

int contains_int(int *arr, size_t n, int val)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (arr[i] == val)
            return 1;
    }
    return 0;
}

int is_row_part_of_testing_fold(int row, const ModelContext *ctx)
{
    size_t lower_bound = ctx->testingFoldIdx * ctx->rowsPerFold;
    size_t upper_bound = lower_bound + ctx->rowsPerFold;

    if (row >= lower_bound && row <= upper_bound)
        return 1;
    else
        return 0;
}

double **combine_arrays(double **first, double **second, size_t n1, size_t n2, size_t cols)
{
    double **combined = (double **)malloc((n1 + n2) * sizeof(double) * cols);
    int row_index = 0;
    for (size_t i = 0; i < n1; ++i)
    {
        double *row = first[i];
        combined[row_index++] = row;
    }
    for (size_t j = 0; j < n2; ++j)
    {
        double *row = second[j];
        combined[row_index++] = row;
    }
    return combined;
}

double **_2d_malloc(const size_t rows, const size_t cols)
{
    double **data;
    double *ptr;

    int len = sizeof(double *) * rows + sizeof(double) * cols * rows;
    data = (double **)malloc(len);

    ptr = (double *)(data + rows);

    for (size_t i = 0; i < rows; ++i)
        data[i] = ptr + cols * i;

    return data;
}

double **_2d_calloc(const size_t rows, const size_t cols)
{
    double **data;
    double *ptr;

    int len = sizeof(double *) * rows + sizeof(double) * cols * rows;
    data = (double **)calloc(len, sizeof(double));

    ptr = (double *)(data + rows);

    for (size_t i = 0; i < rows; ++i)
        data[i] = ptr + cols * i;

    return data;
}

double _1d_checksum(double *data, size_t size)
{
    double sum = 0;
    for (size_t i = 0; i < size; ++i)
    {
        sum += data[i];
    }
    return sum;
}

double _2d_checksum(double **data, size_t rows, size_t cols)
{
    double sum = 0;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            sum += data[i][j];
        }
    }
    return sum;
}
