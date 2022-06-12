/*
@author andrii dobroshynski
*/

#ifndef tree_h
#define tree_h

#include <float.h>
#include <stdlib.h>
#include "../utils/utils.h"

typedef struct DecisionTreeData DecisionTreeData;
typedef struct DecisionTreeNode DecisionTreeNode;
typedef struct DecisionTreeDataSplit DecisionTreeDataSplit;
typedef struct DecisionTreeTargetClasses DecisionTreeTargetClasses;

/*
Represents a single node in a decision tree that comprise a random forest.
*/
struct DecisionTreeNode
{
    long id;
    struct DecisionTreeNode *leftChild;
    struct DecisionTreeNode *rightChild;

    double *half1;
    double *half2;
    size_t size_half1;
    size_t size_half2;

    double split_value;
    long split_index;
    DecisionTreeData *split_data_halves;

    // if the node is a leaf
    int left_leaf;
    int right_leaf;
};

struct DecisionTreeData
{
    size_t length;
    double **data;
};

struct DecisionTreeDataSplit
{
    int index;
    double value;
    double gini;
    DecisionTreeData *data;
};

struct DecisionTreeTargetClasses
{
    size_t count;
    int *labels;
};

/*
Functions to free memory allocated for the structs.
*/
void free_decision_tree_data(DecisionTreeData *data_split);
void free_decision_tree_node(const DecisionTreeNode *node, long *freeCount);

/*
Creates a new empry DecisionTreeNode with id from the strictly increasing
id generator.
*/
DecisionTreeNode *empty_node(long *id);

/*
Function to recursively grow a DecisionTreeNode by splitting the dataset and creating 
left / right children until fully splitting the rows across all nodes.
*/
void grow(DecisionTreeNode *decision_tree,
          size_t max_depth,
          size_t min_samples_leaf,
          size_t max_features,
          int depth,
          size_t rows,
          size_t cols,
          long *nodeId,
          const ModelContext *ctx);

/*
Calculates the best split for the 'data' given a number of randomly selected features from the data
(columns) up to the number of maximum number of features 'max_features'.
*/
DecisionTreeDataSplit calculate_best_data_split(double **data,
                                                size_t max_features,
                                                size_t rows,
                                                size_t cols,
                                                const ModelContext *ctx);

/*
Populates a given DecisionTreeNode with data from the DecisionTreeDataSplit struct 
pointed to by 'data_split'.
*/
void populate_split_data(DecisionTreeNode *node, DecisionTreeDataSplit *data_split);
/*
Given a row of data and a trained decision tree, computes the predicted class target value for the row 
and writes the prediction into the variable pointed to 'prediction_val'.
*/
void make_prediction(const DecisionTreeNode *decision_tree, double *row, int *prediction_val);

#endif // tree_h
