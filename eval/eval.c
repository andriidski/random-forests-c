//
// @author andrii dobroshynskyi
//

#include <stdlib.h>
#include "eval.h"

//
// runs a grid search over the random forest fitted on the data
//
// selects the best parameters based on a cross-validation accuracy score
// from a list of parameters to choose from
//
struct RF_params grid_search(float** data, struct dim data_dim)
{
    int n = 3;

    // init the options for number of trees to: 10, 100, 1000
    int* n_estimators = malloc(sizeof(int) * n);
    n_estimators[0] = 10;
    n_estimators[1] = 50;
    n_estimators[2] = 100;

    // init the options for the max depth for a tree to: 3, 7, 10
    int* max_depths = malloc(sizeof(int) * n);
    max_depths[0] = 3;
    max_depths[1] = 7;
    max_depths[2] = 10;

    // defaults based on SKLearn's defaults / hand picked in order to compare performance with same parameters
    int max_features = 3;
    int min_samples_leaf = 2;
    int max_depth = 7;
    int ratio = 1; // don't withhold any data when sampling

    // number of folds for cross validation
    int k_folds = 5;

    // best params from grid search
    double best_accuracy = -1;
    int best_n_estimators = -1;

    // grid search across the parameters
    for(int i=0; i < n; i++)
    {
        // variable parameters
        int trees = n_estimators[i];

        struct RF_params params = {n_estimators:trees, max_depth:max_depth, min_samples_leaf:min_samples_leaf, max_features:max_features, sampling_ratio:ratio};
        double accuracy_for_params = cross_validation(data, params, data_dim.rows, data_dim.cols, k_folds);

        // update best accuracy and best parameters found so far from grid search
        if(accuracy_for_params > best_accuracy)
        {
            best_accuracy = accuracy_for_params;
            best_n_estimators = trees;
        }
    }
    // free aux arrays
    free(n_estimators);
    free(max_depths);

    return (struct RF_params){n_estimators:best_n_estimators, max_depth:max_depth, min_samples_leaf:min_samples_leaf, max_features:max_features, sampling_ratio:ratio};
}

//
// k-fold Cross Validation for preliminary evaluation of the random forest on the available test/train data
// splits the data set provided into k equal folds and grows k forests
//
double cross_validation(float** data, struct RF_params params, int rows, int cols, int k_folds)
{
    // data set split into k folds
    float*** folds = k_fold_split(rows, cols, data, k_folds);
    int rows_per_fold = rows / k_folds;

    // score for each grown forest
    double* scores = malloc(k_folds * sizeof(double));

    for(int i=0; i < k_folds; i++)
    {
        float** training = get_training_folds(folds, k_folds, i, rows, cols);
        float** test = folds[i];
        // grow the forest on all folds but one
        struct Node** rf = fit_model(training, params, rows - rows_per_fold, cols);

        // get predictions from the forest
        float* predictions = get_predictions(test, rows_per_fold, rf, params.n_estimators);
        // get actual labels
        float* actual = get_class_labels_from_fold(test, rows_per_fold, cols);
        double accuracy = get_accuracy(rows_per_fold, actual, predictions);
        scores[i] = accuracy;
        
        free(actual);
        free(predictions);
    }
    return average_accuracy(scores, k_folds);
}

float* get_class_labels_from_fold(float** fold, int rows_in_fold, int cols)
{
    float* class_labels = malloc(rows_in_fold * sizeof(float));
    for(int i=0; i < rows_in_fold; i++)
    {
        float* row = fold[i];
        class_labels[i] = row[cols-1];
    }
    return class_labels;
}

float** get_training_folds(float*** folds, int n_folds, int selected_test_fold, int rows, int cols)
{
    int count_per_fold = rows / n_folds;
    float** training_folds = malloc((rows - count_per_fold) * cols * sizeof(float));
    int count = 0;
    for(int i=0; i < n_folds; i++)
    {
        if(i == selected_test_fold) continue;
        float** fold = folds[i];
        for(int j=0; j < count_per_fold; j++)
        {
            training_folds[count] = fold[j];
            count++;
        }
    }
    return training_folds;
}