/*
@author andrii dobroshynski
*/

#include "forest.h"

const DecisionTreeNode *train_model_tree(double **data,
                                         const RandomForestParameters *params,
                                         const struct dim *csv_dim,
                                         long *nodeId /* Ascending node ID generator */,
                                         const ModelContext *ctx)
{
    DecisionTreeNode *root = empty_node(nodeId);
    DecisionTreeDataSplit data_split = calculate_best_data_split(data,
                                                                 params->max_features,
                                                                 csv_dim->rows,
                                                                 csv_dim->cols,
                                                                 ctx);

    if (log_level > 1)
        printf("calculated best split for the dataset in train_model_tree\n"
               "half1: %ld\nhalf2: %ld\nbest gini: %f\nbest value: %f\nbest index: %d\n",
               data_split.data[0].length,
               data_split.data[1].length,
               data_split.gini,
               data_split.value,
               data_split.index);

    populate_split_data(root, &data_split);

    // Start building the tree recursively.
    grow(root,
         params->max_depth,
         params->min_samples_leaf,
         params->max_features,
         1 /* Current depth. */,
         csv_dim->rows,
         csv_dim->cols,
         nodeId,
         ctx);

    // Free any temp memory.
    free(data_split.data);

    return root;
}

const DecisionTreeNode **train_model(double **data,
                                     const RandomForestParameters *params,
                                     const struct dim *csv_dim,
                                     const ModelContext *ctx)
{
    // Random forest model which is stored as a contigious list of pointers to DecisionTreeNode structs.
    const DecisionTreeNode **random_forest = (const DecisionTreeNode **)
        malloc(sizeof(DecisionTreeNode *) * params->n_estimators);

    // Node ID generator. We use this such that every node in the tree gets assigned a strictly
    // increasing ID for debugging.
    long nodeId = 0;

    // Populate the array with allocated memory for the random forest with pointers to individual decision
    // trees.
    for (size_t i = 0; i < params->n_estimators; ++i)
    {
        random_forest[i] = train_model_tree(data, params, csv_dim, &nodeId, ctx);
    }
    return random_forest;
}

int predict_model(const DecisionTreeNode ***random_forest, size_t n_estimators, double *row)
{
    int zeroes = 0;
    int ones = 0;
    for (size_t i = 0; i < n_estimators; ++i)
    {
        int prediction;
        make_prediction((*random_forest)[i] /* root of the tree */,
                        row,
                        &prediction);

        if (prediction == 0)
            zeroes++;
        else if (prediction == 1)
            ones++;
        else
        {
            printf("Error: currently only support binary classification, i.e. prediction values 0/1, got: %d\n",
                   prediction);
            exit(1);
        }
    }
    if (ones > zeroes)
        return 1;
    else
        return 0;
}

void free_random_forest(const DecisionTreeNode ***random_forest, const size_t length)
{
    long freeCount = 0;
    for (size_t idx = 0; idx < length; ++idx)
    {
        // Recursively free this DecisionTree rooted at the current node.
        free_decision_tree_node((*random_forest)[idx], &freeCount);
    }
    // Free the actual array of pointers to the nodes.
    free(*random_forest);

    if (log_level > 2)
        printf("total DecisionTreeNode freed: %ld\n", freeCount);
}

void print_params(const RandomForestParameters *params)
{
    printf("using RandomForestParameters:\n  n_estimators: %ld\n  max_depth: %ld\n  min_samples_leaf: %ld\n  max_features: %ld\n",
           params->n_estimators,
           params->max_depth,
           params->min_samples_leaf,
           params->max_features);
}
