/*
@author andrii dobroshynski
*/

#ifndef forest_h
#define forest_h

#include <stdlib.h>
#include "tree.h"

extern int log_level;

/*
Parameters for a Random Forest model.
*/
struct RandomForestParameters
{
    size_t n_estimators;     // Number of trees in a forest.
    size_t max_depth;        // Maximum depth of a tree.
    size_t min_samples_leaf; // Minimum number of data samples at a leaf node.
    size_t max_features;     // Number of features considered when calculating the best data split.
};

typedef struct RandomForestParameters RandomForestParameters;

/*
Function to print a RandomForestParameters struct for debugging.
*/
void print_params(const RandomForestParameters *params);

/*
Trains a single decision tree on the provided data and returns a pointer to the root DecisionTreeNode
of the tree stored on the heap.
*/
const DecisionTreeNode *
train_model_tree(double **data,
                 const RandomForestParameters *params,
                 const struct dim *csv_dim,
                 long *nodeId /* Ascending node ID generator */,
                 const ModelContext *ctx);

/*
Trains a random forest model that is comprised of individually built decision trees. Returns an array 
of pointers to DecisionTreeNode's that are the roots of the decision trees in the random forest model.
*/
const DecisionTreeNode **train_model(double **data,
                                     const RandomForestParameters *params,
                                     const struct dim *csv_dim,
                                     const ModelContext *ctx);

/*
Given a single row, gets predictions from every decision tree in the 'random_forest' model
for the class target that the row should be classified into and returns the class target value
that is the majority vote.
*/
int predict_model(const DecisionTreeNode ***random_forest, size_t n_estimators, double *row);

/*
Frees memory for a given random forest model (array of pointers to DecisionTreeNode's).
*/
void free_random_forest(const DecisionTreeNode ***random_forest, const size_t length);

#endif // forest_h
