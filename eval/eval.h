/*
@author andrii dobroshynski
*/

#ifndef eval_h
#define eval_h

#include "../model/tree.h"
#include "../model/forest.h"
#include "../utils/utils.h"
#include "../utils/data.h"

/*
Runs a hyperparameter search across a number of pre-defined parameters for the random forest model and
reports the best parameters. Calls 'cross_validate' on each parameter configuration to get the cross validation
accuracy for each set-up. Can be adjusted to run across as many parameters as needed.
*/
void hyperparameter_search(double **data, struct dim *csv_dim);

/*
Runs k-fold cross validation on the 'data' and returns the accuracy. In the process builds up a random
forest model for each iteration and evaluates on a separate test fold.
*/
double cross_validate(double **data,
                      const RandomForestParameters *params,
                      const struct dim *csv_dim,
                      const int k_folds);

#endif // eval_h
