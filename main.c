/*
@author andrii dobroshynski
*/

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "eval/eval.h"
#include "utils/argparse.h"
#include "utils/data.h"
#include "utils/utils.h"

/* Our argp parser. */
static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char **argv)
{
    struct arguments arguments;

    // Default argument values.
    arguments.log_level = 1;
    arguments.rows = 0;
    arguments.cols = 0;

    /* Parse our arguments; every option seen by parse_opt will
     be reflected in arguments. */
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    // Set the log level to whatever was parsed from the arguments or the default value.
    set_log_level(arguments.log_level);

    // Optionally set the random seed if a specific random seed was provided via an argument.
    if (arguments.random_seed)
        srand(arguments.random_seed);
    else
        srand(time(NULL));

    // Read the csv file from args which must be parsed now.
    const char *file_name = arguments.args[0];

    // If the values for rows and cols were provided as arguments, then use them for the
    // 'dim' struct, otherwise call 'parse_csv_dims()' to parse the csv file provided to
    // compute the size of the csv file.
    struct dim csv_dim;

    if (arguments.rows && arguments.cols)
        csv_dim = (struct dim){rows : arguments.rows, cols : arguments.cols};
    else
        csv_dim = parse_csv_dims(file_name);

    if (log_level > 0)
        printf("using:\n  verbose log level: %d\n  rows: %ld, cols: %ld\nreading from csv file:\n  \"%s\"\n",
               log_level,
               csv_dim.rows,
               csv_dim.cols,
               file_name);

    // Allocate memory for the data coming from the .csv and read in the data.
    double *data = malloc(sizeof(double) * csv_dim.rows * csv_dim.cols);
    parse_csv(file_name, &data, csv_dim);

    // Compute a checksum of the data to verify that loaded correctly.
    if (log_level > 1)
        printf("data checksum = %f\n", _1d_checksum(data, csv_dim.rows * csv_dim.cols));

    const int k_folds = 1;

    if (log_level > 0)
        printf("using:\n  k_folds: %d\n", k_folds);

    // Example configuration for a random forest model.
    const RandomForestParameters params = {
        n_estimators : 3 /* Number of trees in the random forest model. */,
        max_depth : 7 /* Maximum depth of a tree in the model. */,
        min_samples_leaf : 3,
        max_features : 3
    };

    // Print random forest parameters.
    if (log_level > 0)
        print_params(&params);

    // Pivot the csv file data into a two dimensional array.
    double **pivoted_data;
    pivot_data(data, csv_dim, &pivoted_data);

    if (log_level > 1)
        printf("checksum of pivoted 2d array: %f\n", _2d_checksum(pivoted_data, csv_dim.rows, csv_dim.cols));

    // Start the clock for timing.
    clock_t begin_clock = clock();

    double cv_accuracy = cross_validate(pivoted_data, &params, &csv_dim, k_folds);
    printf("cross validation accuracy: %f%% (%ld%%)\n",
           (cv_accuracy * 100),
           (long)(cv_accuracy * 100));

    // Record and output the time taken to run.
    clock_t end_clock = clock();
    printf("(time taken: %fs)\n", (double)(end_clock - begin_clock) / CLOCKS_PER_SEC);

    // Free loaded csv file data.
    free(data);
    free(pivoted_data);
}
