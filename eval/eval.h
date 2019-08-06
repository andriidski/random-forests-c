//
// @author andrii dobroshynskyi
//

#ifndef RANDOM_FOREST_C_EVAL_H
#define RANDOM_FOREST_C_EVAL_H

#include "../model/model.h"
#include "../utils/utils.h"

float* get_class_labels_from_fold(float** fold, int rows_in_fold, int cols);
float** get_training_folds(float*** folds, int n_folds, int selected_test_fold, int rows, int cols);

struct RF_params grid_search(float** data, struct dim data_dim);
double cross_validation(float** data, struct RF_params params, int rows, int cols, int k_folds);

#endif //RANDOM_FOREST_C_EVAL_H
