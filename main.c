//
// @author andrii dobroshynskyi
//

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "utils/utils.h"
#include "eval/eval.h"

#define BUFFER_SIZE 1024

struct dim get_csv_dimensions(FILE *file);
float** read_data(FILE *csv, float*** data, struct dim csv_dim);

//
// parse a csv to get the row / column dimensions
//
struct dim get_csv_dimensions(FILE *file)
{
    const char *delimeter = ",";
    char buffer[ BUFFER_SIZE ];
    char *token;

    int rows_count = 0;
    int cols_count = 0;

    while(fgets(buffer, BUFFER_SIZE, file) != NULL){
        token = strtok(buffer, delimeter);
        while(token != NULL) {
            if(strstr(token, "\n") != NULL) {
                rows_count++;
                if(cols_count != 0) {
                    if(verbose) printf("\ncols: %d\n", cols_count);
                    cols_count = 0;
                }
            }
            cols_count++;
            if(verbose) printf("token: %s\n", token);
            token = strtok(NULL, delimeter);
        }
    }
    rows_count += 1;
    cols_count -= 1;
    if(verbose) ("\nrows counted: %d\ncols counted: %d\n",rows_count, cols_count);
    fseek(file, 0, SEEK_SET);

    return (struct dim){rows:rows_count, cols:cols_count};
}

//
// parse the input csv file into memory
//
float** read_data(FILE *csv, float*** data, struct dim csv_dim)
{
    float temp;
    int i;
    int j;

    if(verbose) printf("\nloading data from csv...\n\n");
    for(i = 0; i < csv_dim.rows; i++) {
        for(j = 0; j < csv_dim.cols; j++) {
            fscanf(csv, "%f", &temp);
            (*data)[i][j] = temp;
            fscanf(csv, ",");
            if(verbose) printf("%f ", (*data)[i][j]);
        }
        if(verbose) printf("\n");
    }
    fclose(csv);
}

int main()
{
    srand(time(NULL));

    // open csv file
    char* fn = "../data.csv";
    FILE *csv_file;
    csv_file = fopen(fn,"r");
    if(csv_file == NULL) {
        printf("Error: can't open file \n");
        return -1;
    }

    // get dimensions of csv file
    struct dim csv_dim = get_csv_dimensions(csv_file);
    if(verbose) printf("dim | rows: %d\n", csv_dim.rows);
    if(verbose) printf("dim | cols: %d\n", csv_dim.cols);

    int rows = csv_dim.rows;
    int cols = csv_dim.cols;

    float** data = create_array_2d(rows,cols);

    // read in the data into memory
    read_data(csv_file, &data, csv_dim);

    int k_folds = 5;
    struct RF_params params = {n_estimators:10, max_depth:7, min_samples_leaf:3, max_features:3, sampling_ratio:0.9};

    // start clock for timing
    clock_t begin = clock();

    // train and test at the same time
    double cv_accuracy = cross_validation(data, params, rows, cols, k_folds);

    // end clock for timing
    clock_t end = clock();
    printf("\ntime taken: %fs | accuracy: %.20f\n", (double)(end - begin) / CLOCKS_PER_SEC, cv_accuracy);

    destroy_array_2d(data);
}