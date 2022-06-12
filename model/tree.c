/*
@author andrii dobroshynski
*/

#include "tree.h"

/*
Allocates memory for an empty DecisionTreeNode and returns a pointer to the node.
*/
DecisionTreeNode *empty_node(long *id)
{
    DecisionTreeNode *node = malloc(sizeof(DecisionTreeNode));

    node->id = (*id);
    node->leftChild = NULL;
    node->rightChild = NULL;

    node->half1 = NULL;
    node->half2 = NULL;
    node->size_half1 = 0;
    node->size_half2 = 0;

    node->split_index = -1;
    node->split_value = -1;
    node->split_data_halves = NULL;

    (*id)++;

    if (log_level > 2)
        printf("created a DecisionTreeNode with id %ld stored at address %p \n", node->id, node);

    return node;
}

/*
Populates a given DecisionTreeNode with data from the DecisionTreeDataSplit struct 
pointed to by 'data_split'.
*/
void populate_split_data(DecisionTreeNode *node, DecisionTreeDataSplit *data_split)
{
    node->split_index = (*data_split).index;
    node->split_value = (*data_split).value;
    node->split_data_halves = (*data_split).data;
}

/*
Given a two dimensional array of data finds and returns a DecisionTreeTargetClasses
struct with unique target classes found in the dataset at column with index 'cols - 1'.
*/
DecisionTreeTargetClasses get_target_class_values(double **data, size_t rows, size_t cols, const ModelContext *ctx)
{
    if (log_level > 1)
        printf("generating class value set...\n");

    size_t count = 0;
    int *target_class_values = malloc(count * sizeof(int));

    for (size_t i = 0; i < rows; ++i)
    {
        // Skip rows that we are withholding from training for evaluation.
        if (is_row_part_of_testing_fold(i, ctx))
        {
            if (log_level > 1)
                printf("  skipping row %ld which is part of testing fold %ld\n", i, ctx->testingFoldIdx);
            continue;
        }

        int class_target = (int)data[i][cols - 1];
        if (!contains_int(target_class_values, count, class_target))
        {
            if (log_level > 1)
                printf("adding %d \n", class_target);
            count++;
            int *temp = realloc(target_class_values, count * sizeof(int));
            if (temp != NULL)
                target_class_values = temp;
            target_class_values[count - 1] = class_target;
        }
    }
    if (log_level > 1)
        printf("-------------------------------\ncount of unique classes: %ld\n", count);
    return (DecisionTreeTargetClasses){count, target_class_values};
}

/*
Given a two dimensional array of data returns the leaf node class value for the given
data. The leaf node class value is whichever class value that is the class target value for
the majority of the rows in the data.
*/
int get_leaf_node_class_value(double **data, size_t rows, size_t cols)
{
    int zeroes = 0;
    int ones = 0;
    for (size_t i = 0; i < rows; ++i)
    {
        int class_label = (int)data[i][cols - 1];
        if (class_label == 0)
            zeroes++;
        else if (class_label == 1)
            ones++;
        else
        {
            printf("Error: currently only support binary classification, i.e. class target values 0/1, got: %d\n",
                   class_label);
            exit(1);
        }
    }
    if (ones >= zeroes)
        return 1;
    else
        return 0;
}

/*
Given a two dimensional array of data and parameters for a split, splits the data into two halves and
returns a pointer to an array of two DecisionTreeData for the two halves of the split.
*/
DecisionTreeData *split_dataset(int feature_index,
                                double value,
                                double **data,
                                size_t rows,
                                size_t cols)
{
    if (log_level > 1)
        printf("splitting dataset into two halves...\n");

    // Buffers to hold rows of data as we are distributing rows based on the split.
    double **left = (double **)malloc(1 * sizeof(double) * cols);
    double **right = (double **)malloc(1 * sizeof(double) * cols);

    size_t left_count = 0;
    size_t right_count = 0;

    for (size_t i = 0; i < rows; ++i)
    {
        double *row = data[i];
        if (row[feature_index] < value)
        {
            // Copy the row into the left half and resize the buffer.
            left[left_count++] = row;
            double **temp = realloc(left, left_count * sizeof(double) * cols);
            if (temp != NULL)
                left = temp;
        }
        else
        {
            // Copy the row into the right half and resize the buffer.
            right[right_count++] = row;
            double **temp = realloc(right, right_count * sizeof(double) * cols);
            if (temp != NULL)
                right = temp;
        }
    }
    DecisionTreeData *data_split = malloc(sizeof(DecisionTreeData) * 2);
    data_split[0] = (DecisionTreeData){left_count, left};
    data_split[1] = (DecisionTreeData){right_count, right};

    if (log_level > 1)
        printf("split dataset into: %ld | %ld\n", left_count, right_count);

    return data_split;
}

double calculate_gini_index(DecisionTreeData *data_split,
                            int *class_labels,
                            size_t class_labels_count,
                            size_t cols)
{
    if (log_level > 1)
        printf("calculating gini index based on split...\n");

    // DecisionTreeData data split should consist of two halves.
    int count = 2;
    size_t n_instances = data_split[0].length + data_split[1].length;
    double gini = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        DecisionTreeData group = data_split[i];

        size_t size = group.length;
        if (size == 0)
            continue;

        double sum = 0.0;
        for (size_t j = 0; j < class_labels_count; ++j)
        {
            int class = class_labels[j];
            int occurences = 0;
            for (size_t k = 0; k < size; ++k)
            {
                int label = (int)group.data[k][cols - 1];
                if (label == class)
                    occurences += 1;
            }
            double p_class = (double)occurences / (double)size;
            sum += (p_class * p_class);
        }
        gini += (1.0 - sum) * ((double)size / (double)n_instances);
    }

    if (log_level > 1)
    {
        printf("gini: %f\n", gini);
        printf("-----------------------------------------\n");
    }

    return gini;
}

DecisionTreeDataSplit calculate_best_data_split(double **data,
                                                size_t max_features,
                                                size_t rows,
                                                size_t cols,
                                                const ModelContext *ctx)
{
    if (log_level > 1)
    {
        printf("calculating best split for dataset...\n");
        printf("rows: %ld\ncols: %ld\n", rows, cols);
    }

    // Target classes available in this dataset.
    DecisionTreeTargetClasses classes = get_target_class_values(data, rows, cols, ctx);

    // Keeping track of best data split available along with best parameters associated with
    // that data split.
    DecisionTreeData *best_data_split = NULL;
    double best_value = DBL_MAX;
    double best_gini = DBL_MAX;
    int best_index = INT_MAX;

    // Create a features array and initialize to avoid non-set memory.
    int *features = malloc(max_features * sizeof(int));
    for (size_t i = 0; i < max_features; ++i)
        features[i] = -1;

    size_t count = 0;
    while (count < max_features)
    {
        // Maximum index for a feature which should not include the class target column index
        // which is 'cols - 1'.
        int max = cols - 2;
        int min = 0;
        int index = rand() % (max + 1 - min) + min;
        if (!contains_int(features, max_features /* size of 'features' array */, index))
        {
            if (log_level > 1)
                printf("adding unique index: %d\n", index);
            features[count++] = index;
        }
    }
    if (log_level > 1)
        printf("-----------------------------------------\n");

    for (size_t i = 0; i < max_features; ++i)
    {
        int feature_index = features[i];
        for (size_t j = 0; j < rows; ++j)
        {
            DecisionTreeData *data_split = split_dataset(feature_index,
                                                         data[j][feature_index],
                                                         data,
                                                         rows,
                                                         cols);
            double gini = calculate_gini_index(data_split, classes.labels, classes.count, cols);

            if (gini < best_gini)
            {
                best_index = feature_index;
                best_value = data[j][feature_index];
                best_gini = gini;

                // First free the memory that was previously allocated for the 'data_split' and pointer
                // which was assigned to 'best_data_split' since now we have found a new better split.
                if (best_data_split)
                    free_decision_tree_data(best_data_split);

                best_data_split = data_split;
            }
            else
            {
                free_decision_tree_data(data_split);
            }
        }
    }

    // Free any other memory.
    free(features);
    free(classes.labels);

    return (DecisionTreeDataSplit){best_index, best_value, best_gini, best_data_split};
}

void grow(DecisionTreeNode *decision_tree,
          size_t max_depth,
          size_t min_samples_leaf,
          size_t max_features,
          int depth,
          size_t rows,
          size_t cols,
          long *nodeId,
          const ModelContext *ctx)
{
    DecisionTreeData left_half = decision_tree->split_data_halves[0];
    DecisionTreeData right_half = decision_tree->split_data_halves[1];

    double **left = left_half.data;
    double **right = right_half.data;

    decision_tree->split_data_halves = NULL;

    if (left == NULL || right == NULL)
    {
        // If we are at the leaf node, then combine both left and right side data and compute get the leaf
        // node class for the combined data.
        double **combined_data = combine_arrays(left, right, left_half.length, right_half.length, cols);
        int leaf = get_leaf_node_class_value(combined_data, rows, cols);

        decision_tree->left_leaf = leaf;
        decision_tree->right_leaf = leaf;

        free(left);
        free(right);
        free(combined_data);

        return;
    }
    if (depth >= max_depth)
    {
        decision_tree->left_leaf = get_leaf_node_class_value(left, left_half.length /* rows */, cols);
        decision_tree->right_leaf = get_leaf_node_class_value(right, right_half.length /* rows */, cols);

        free(left);
        free(right);

        return;
    }
    if (left_half.length <= min_samples_leaf)
    {
        decision_tree->left_leaf = get_leaf_node_class_value(left, left_half.length /* rows */, cols);
    }
    else
    {
        DecisionTreeDataSplit data_split = calculate_best_data_split(left,
                                                                     max_features,
                                                                     left_half.length /* rows */,
                                                                     cols,
                                                                     ctx);

        // Create the left child of the current node and populate with data from the data split.
        decision_tree->leftChild = empty_node(nodeId);
        populate_split_data(decision_tree->leftChild, &data_split);

        grow(decision_tree->leftChild,
             max_depth,
             min_samples_leaf,
             max_features,
             depth + 1 /* since we are now at the next 'level' in the tree */,
             rows,
             cols,
             nodeId,
             ctx);

        free(data_split.data);
    }
    if (right_half.length <= min_samples_leaf)
    {
        decision_tree->right_leaf = get_leaf_node_class_value(right, right_half.length, cols);
    }
    else
    {
        DecisionTreeDataSplit data_split = calculate_best_data_split(right,
                                                                     max_features,
                                                                     right_half.length /* rows */,
                                                                     cols,
                                                                     ctx);

        // Create the right child of the current node and populate with data from the data split.
        decision_tree->rightChild = empty_node(nodeId);
        populate_split_data(decision_tree->rightChild, &data_split);

        grow(decision_tree->rightChild,
             max_depth,
             min_samples_leaf,
             max_features,
             depth + 1 /* since we are now at the next 'level' in the tree */,
             rows,
             cols,
             nodeId,
             ctx);

        free(data_split.data);
    }

    free(left);
    free(right);
}

void make_prediction(const DecisionTreeNode *decision_tree, double *row, int *prediction_val)
{
    if (row[decision_tree->split_index] < decision_tree->split_value)
    {
        if (decision_tree->leftChild != NULL)
            make_prediction(decision_tree->leftChild, row, prediction_val);
        else
            (*prediction_val) = decision_tree->left_leaf;
    }
    else
    {
        if (decision_tree->rightChild != NULL)
            make_prediction(decision_tree->rightChild, row, prediction_val);
        else
            (*prediction_val) = decision_tree->right_leaf;
    }
}

/*
Frees memory for a given DecisionTreeNode.
*/
void free_decision_tree_node(const DecisionTreeNode *node, long *freeCount)
{
    (*freeCount)++;
    if (node->leftChild)
        free_decision_tree_node(node->leftChild, freeCount);
    if (node->rightChild)
        free_decision_tree_node(node->rightChild, freeCount);

    if (log_level > 2)
        printf("freeing DecisionTreeNode with id=%ld\n", node->id);

    if (node && node->split_data_halves && node->split_data_halves->length)
    {
        free(node->split_data_halves[0].data);
        free(node->split_data_halves[1].data);
        free(node->split_data_halves);
    }

    free((void *)node);
}

/*
Frees memory for a given DecisionTreeData.
*/
void free_decision_tree_data(DecisionTreeData *data_split)
{
    free(data_split[0].data);
    free(data_split[1].data);
    free(data_split);
}
