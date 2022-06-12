# Random Forests - C
Basic implementation of random forests and accompanying decision trees in C

## Running the code
### Training
A model can be trained on a dataset with the `fit_model()` function that takes the following parameters:
- `float** training_data` - a 2D matrix of training data (equivalent to a DataFrame in Python)
- `struct RF_params params` - a custom struct that holds the parameters to customize the Random Forest
- `int rows`, `int cols` - row x col dimensions of the training data matrix

Example:

`struct RF_params params = {n_estimators:10, max_depth:7, min_samples_leaf:3, max_features:3, sampling_ratio:1};`

`struct Node** model = fit_model(data, params, rows, cols);`
u
Training a Random Forest model of 10 estimators (trees), with a max depth of 7 for a single decision tree, min features on split of 3. The last parameter is used for growing a single decision tree when we want to omit some data when sampling.

### Predicting

After training, predictions can be made with the `get_predictions()` function that takes the following parameters:
- `float** test_data` - dataset to predict on
- `int rows_test` - the number of rows in the dataset that we are predicting for
- `struct Node** trees` - the trained Random Forest model
- `int n_estimators` - number of estimators (trees) in the trained Random Forest model

Additionally, true class values can be extracted from the training dataset with the function `get_class_labels_from_fold()`, 
and the accuracy of the predictions can be computed with a utility function `get_accuracy()` 

Example:

```
float* predictions = get_predictions(data, rows, model, params.n_estimators);
float* actual = get_class_labels_from_fold(data, rows, cols);

double accuracy = get_accuracy(rows, actual, predictions);
printf("\naccuracy: %.20f\n",accuracy);
```


Making predictions with the model trained in previous step (Training), and evaluating the accuracy of the predictions


## Project structure
The project is organized into:
- model (RFs and decision trees)
- evaluation code for performing tasks like grid search or cross-validation
- utilities for data management

## Reference
Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.