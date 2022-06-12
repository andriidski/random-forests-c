/*
@author andrii dobroshynski
*/

#include "data.h"

struct dim parse_csv_dims(const char *file_name)
{
    FILE *csv_file;
    csv_file = fopen(file_name, "r");

    if (csv_file == NULL)
    {
        printf("Error: can't open file: %s\n", file_name);
        exit(-1);
    }

    const char *delimiter = ",";

    char *buffer = malloc(BUFSIZ);
    char *token;

    // Keeping track of how many rows and columns there are.
    int rows = 0;
    int cols = 0;

    // Reach each line of the file into the buffer.
    while (fgets(buffer, BUFSIZ, csv_file) != NULL)
    {
        ++rows;

        // We use a second counter for columns in order to count the number of columns
        // for each row we read and will trigger an assert if there is a mismatch in the
        // size of the cols, i.e. not all rows have the same number of columns which would
        // cause undefined behavior.
        int curr_cols = 0;

        // Get every token and print it.
        token = strtok(buffer, delimiter);
        while (token != NULL)
        {
            ++curr_cols;
            // printf("%s\n", token);

            // Get the next token.
            token = strtok(NULL, delimiter);
        }
        if (cols == 0)
        {
            cols = curr_cols;
        }
        else
        {
            assert(curr_cols == cols && "Error: every row must have the same amount of columns");
        }
    }
    // We read one extra row for the csv header, so adjust here.
    --rows;

    fclose(csv_file);
    free(buffer);

    // Make sure that the dimensions are valid.
    assert(rows > 0 && "# of rows in csv must be > 0");
    assert(cols > 0 && "# of cols in csv must be > 0");

    return (struct dim){rows : rows, cols : cols};
}

void parse_csv(const char *file_name, double **data_p, const struct dim csv_dim)
{
    FILE *csv_file;
    csv_file = fopen(file_name, "r");

    if (csv_file == NULL)
    {
        printf("Error: can't open file: %s\n", file_name);
        exit(-1);
    }

    const char *delimiter = ",";

    char *buffer = malloc(BUFSIZ);
    char *token;

    // Keeping track which row we are on.
    int row = 0;
    // Keeping track of the index in the data array.
    int idx = 0;

    // Reach each line of the file into the buffer.
    while (row <= csv_dim.rows && fgets(buffer, BUFSIZ, csv_file) != NULL)
    {
        if (++row == 1)
            continue;

        // Get every token and print it.
        token = strtok(buffer, delimiter);
        while (token != NULL)
        {
            // if (1)
            // printf("%s\n", token);

            (*data_p)[idx] = atof(token);
            // printf("%f\n", (*data)[idx]);
            ++idx;

            // Get the next token.
            token = strtok(NULL, delimiter);
        }
    }

    if (log_level > 1)
        printf("read %d rows from file %s\n", row - 1, file_name);

    fclose(csv_file);
    free(buffer);
}

void pivot_data(double *data, const struct dim csv_dim, double ***pivoted_data_p)
{
    (*pivoted_data_p) = _2d_calloc(csv_dim.rows, csv_dim.cols);

    for (size_t i = 0; i < csv_dim.rows; ++i)
        for (size_t j = 0; j < csv_dim.cols; ++j)
            (*pivoted_data_p)[i][j] = data[(i * csv_dim.cols) + j];
}
