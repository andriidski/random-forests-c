# Random Forests - C
A proof of concept basic implementation of random forests for classification and accompanying decision trees in C.

## Running the code

Fastest way to start experimenting is to
- (1) run the `data.py` script to generate some random CSV data
- (2) compile as preferred (optionally using the `CMakeLists.txt` provided)
- (3) run `./random-forests-c <path_to_csv_file>` or `./random-forests-c --help` to see which arguments are available to configure.

The [`main.c`](./main.c) file contains an example configuration of a random forest and code to run `cross_validate()` which will both train and evaluate a model.

### Training

The `cross_validate()` function runs k-fold cross validation on whatever data is provided -- first trains the model and then evaluates it on every of the testing folds.

The main function that handles model training is `train_model()`
```c
const DecisionTreeNode **train_model(double **data,
                                     const RandomForestParameters *params,
                                     const struct dim *csv_dim,
                                     const ModelContext *ctx);
```
It returns an array of `DecisionTreeNode` pointers to roots of decision trees comprising the forest, and the parameters are

- `**training_data` - training data (equivalent to a DataFrame in Python).
- `*params` - pointer to struct that holds the configuration of a random forest model.
- `*csv_dim` - pointer to a struct holding row x col dimensions of the read data.
- `*ctx` - pointer to a context object that holds some optional data that can be used for training / evaluation.

For example:
```c
const ModelContext ctx = (ModelContext){
    testingFoldIdx : foldIdx /* Fold to use for evaluation. */,
    rowsPerFold : csv_dim->rows / k_folds /* Number of rows per fold. */
};

const DecisionTreeNode **random_forest = (const DecisionTreeNode **)train_model(
    data,
    params,
    csv_dim,
    &ctx);
```

### Evaluation

After training we can evaluate the model with `eval_model()` which returns an accuracy measure for model performance.
For example:

```c
// Evaluate the model that was just trained. We use the fold identified by 'foldIdx' in 'ctx' to evaluate the model.
double accuracy = eval_model(
    random_forest /* Model to evaluate. */,
    data,
    params,
    csv_dim,
    &ctx);
```

## Code structure

- `model` -- random forest and decision trees.
- `eval` -- evaluation code for running `cross_validate()` or `hyperparameter_search()` to test the model.
- `utils` -- utilities for data management, argument parsing, etc.

The optional arguments to the program (can be viewed by running with a `--help` flag)
```
  -c, --num_cols=number      Optional number of cols in the input CSV_FILE, if
                             known
  -r, --num_rows=number      Optional number of rows in the input CSV_FILE, if
                             known
  -l, --log_level=number     Optional debug logging level [0-3]. Level 0 is no
                             output, 3 is most verbose. Defaults to 1.
  -s, --seed=number          Optional random number seed.
```

## Reference
Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.