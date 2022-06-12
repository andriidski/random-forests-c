/*
@author andrii dobroshynski
*/

#include <stdio.h>
#include <stdlib.h>
#include "eval.h"

void hyperparameter_search(double **data, struct dim *csv_dim)
{
    // Init the options for number of trees to: 10, 100, 1000.
    size_t n = 3;

    size_t *estimators = malloc(sizeof(size_t) * n);
    estimators[0] = 10;
    estimators[1] = 50;
    estimators[2] = 100;

    // Init the options for the max depth for a tree to: 3, 7, 10.
    size_t *max_depths = malloc(sizeof(size_t) * n);
    max_depths[0] = 3;
    max_depths[1] = 7;
    max_depths[2] = 10;

    // Defaults based on SKLearn's defaults / hand picked in order to compare performance
    // with the same parameters.
    size_t max_features = 3;
    size_t min_samples_leaf = 2;
    size_t max_depth = 7;

    // Number of folds for cross validation.
    size_t k_folds = 5;

    // Best params computed from running the hyperparameter search.
    size_t best_n_estimators = -1;
    double best_accuracy = -1;

    for (size_t i = 0; i < n; ++i)
    {
        size_t n_estimators = estimators[i]; /* Number of trees in the forest. */

        for (size_t j = 0; j < n; ++j)
        {
            size_t max_depth = max_depths[j];

            RandomForestParameters params = {
                n_estimators : n_estimators,
                max_depth : max_depth,
                min_samples_leaf : min_samples_leaf,
                max_features : max_features
            };

            if (log_level > 0)
            {
                printf("[hyperparameter search] running cross_validate\n");
                printf("[hyperparameter search] ");
                print_params(&params);
            }

            double cv_accuracy = cross_validate(data,
                                                &params,
                                                csv_dim,
                                                k_folds);

            if (log_level > 0)
                printf("[hyperparameter search] cross validation accuracy: %f%% (%ld%%)\n",
                       (cv_accuracy * 100),
                       (long)(cv_accuracy * 100));

            // Update best accuracy and best parameters found so far from the hyperparameter search.
            if (cv_accuracy > best_accuracy)
            {
                best_accuracy = cv_accuracy;
                best_n_estimators = n_estimators;
            }
        }
    }

    // Free auxillary buffers.
    free(estimators);
    free(max_depths);

    printf("[hyperparameter search] run complete\n  best_accuracy: %f\n  best_n_estimators (trees): %ld\n",
           best_accuracy, best_n_estimators);
}

double eval_model(const DecisionTreeNode **random_forest,
                  double **data,
                  const RandomForestParameters *params,
                  const struct dim *csv_dim,
                  const ModelContext *ctx)
{
    // Keeping track of how many predictions have been correct. Accuracy can be
    // computed with 'num_correct' / 'rowsPerFold' (or how many predictions we make).
    long num_correct = 0;

    // Since we are evaluating the model on a single fold (to control overfitting), we start
    // iterating the rows for which we are getting predictions at an offset that can be computed
    // as 'testingFoldIdx * rowsPerFold' and make predictions for 'rowsPerFold' number of rows
    size_t row_id_offset = ctx->testingFoldIdx * ctx->rowsPerFold;
    for (size_t row_id = row_id_offset; row_id < row_id_offset + ctx->rowsPerFold; ++row_id)
    {
        int prediction = predict_model(&random_forest,
                                       params->n_estimators,
                                       data[row_id]);
        int ground_truth = (int)data[row_id][csv_dim->cols - 1];

        if (log_level > 1)
            printf("majority vote: %d | %d ground truth\n", prediction, ground_truth);

        if (prediction == ground_truth)
            ++num_correct;
    }
    return (double)num_correct / (double)ctx->rowsPerFold;
}

double cross_validate(double **data,
                      const RandomForestParameters *params,
                      const struct dim *csv_dim,
                      const int k_folds)
{
    // Sum of all accuracies on every evaluated fold.
    double sumAccuracy = 0;

    // Iterate through the fold indeces and fit models on the selections. The current 'foldIdx' is the index
    // of the fold in the array of all loaded data that is the fold that's currently the test fold, with all of
    // the other folds being used for training.
    for (size_t foldIdx = 0; foldIdx < k_folds; ++foldIdx)
    {
        const ModelContext ctx = (ModelContext){
            testingFoldIdx : foldIdx /* Fold to use for evaluation. */,
            rowsPerFold : csv_dim->rows / k_folds /* Number of rows per fold. */
        };

        // Train an instance of the model with every fold of data except of the fold indentified by
        // 'foldIdx' used for training the the 'foldIdx' fold withheld from training in order to be
        // used for evaluation.
        const DecisionTreeNode **random_forest = (const DecisionTreeNode **)train_model(
            data,
            params,
            csv_dim,
            &ctx);

        // Evaluate the model that was just trained. We use the fold identified by 'foldIdx' to evaluate
        // the model.
        const double accuracy = eval_model(
            random_forest /* Model to evaluate. */,
            data,
            params,
            csv_dim,
            &ctx);
        sumAccuracy += accuracy;

        // Free memory that was used to store the model.
        free_random_forest(&random_forest, params->n_estimators);
    }

    return sumAccuracy / k_folds;
}
