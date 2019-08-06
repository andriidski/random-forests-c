//
// @author andrii dobroshynskyi
//

#include "model.h"
#include <stdio.h>
#include <limits.h>

//
// split the data into an array of multidimensional arrays for k folds
//
float*** k_fold_split(int n, int m, float** data, int k_folds)
{
    if(verbose) printf("\nsplitting data into k folds...\n");
    float*** set_of_folds = (float***) malloc(n*m*k_folds * sizeof(float));

    int offset = 0;
    for(int fold=0; fold < k_folds; fold++)
    {
        if(verbose) printf("\nfold:\n");
        float** current_fold = malloc(n*m * sizeof(float));
        for(int i=0; i < n / k_folds; i++)
        {
            current_fold[i] = data[i+offset];
            for(int j=0; j < m; j++)
            {
                if(verbose) printf("%f ", current_fold[i][j]);
            }
            if(verbose) printf("\n");
        }
        set_of_folds[fold] = current_fold;
        if(verbose) printf("\n-----------------------------------------\n");
        offset += (n / k_folds);
    }
    return set_of_folds;
}

//
// build up a forest of random decision trees based on the given parameters and fit to the training data
//
struct Node** fit_model(float** training_data, struct RF_params params, int rows, int cols)
{
    if(verbose) printf("\ncreating a random forest\nrows: %d\ncols: %d\npercentage of total set: %f\n",rows,cols,params.sampling_ratio);

    struct Node** trees = (struct Node**) malloc(sizeof(struct Node) * params.n_estimators);
    for(int i=0; i < params.n_estimators; i++)
    {
        float** sample = subsample(training_data, params.sampling_ratio, rows, cols);
        int sample_row_count = (int)((float)rows * params.sampling_ratio);

        // sub sampling
        struct Node* tree = build_tree(sample, params.max_depth, params.min_samples_leaf, params.max_features, sample_row_count, cols);
        trees[i] = tree;
    }
    return trees;
}

//
// high level growth of one decision tree
//
struct Node* build_tree(float** training_data, int max_depth, int min_samples_leaf, int max_features, int rows, int cols)
{
    if(verbose) printf("\nabout to build tree\n");
    struct split_params_struct split_params = calculate_best_data_split(training_data, max_features, rows, cols);
    if(verbose) printf("\ncalculated best split for the dataset in build_tree"
                               "\nhalf1: %d\nhalf2: %d\nbest gini: %f\nbest value: %f\nbest index: %d\n",
                       split_params.two_halves[0].length, split_params.two_halves[1].length, split_params.gini, split_params.value, split_params.index);
    struct Node* root = create_node();
    root->two_halves = split_params.two_halves;
    root->index = split_params.index;
    root->value = split_params.value;
    grow(root, max_depth, min_samples_leaf, max_features, 1, rows, cols);
    return root;
}

//
// tree node "constructor"
//
struct Node* create_node()
{
    struct Node* newNode = malloc(sizeof(struct Node));
    newNode->left = NULL;
    newNode->right = NULL;
    newNode->two_halves = NULL;
    newNode->index = -1;
    newNode->value = -1;
    newNode->right_leaf = -1;
    newNode->left_leaf = -1;
    return newNode;
}

//
// return an array of class values (unique classes)
// imitate a hash set
//
struct class_label_struct get_class_values(float** data, int n, int m)
{
    if(verbose) printf("\ngenerating class value set...\n\n");
    int count = 0;
    // create an array to hold class values
    float* class_value_set = malloc(count * sizeof(float));

    for(int i=0; i < n; i++)
    {
        float class_target = data[i][m-1];
        if(!contains_float(class_value_set, count, class_target))
        {
            if(verbose) printf("\nadding %f \n", class_target);
            count++;
            float* temp = realloc(class_value_set, count * sizeof(float));
            if(temp != NULL) class_value_set = temp;
            class_value_set[count-1] = class_target;
        }
    }
    if(verbose) printf("\n-------------------------------\ncount of unique classes: %d\n",count);
    return (struct class_label_struct){count, class_value_set};
}

//
// gets the class label of a leaf node
// currently only implemented for binary classification -> {0,1}
//
float get_leaf_node_class(float** group, int size, int cols)
{
    int zeroes = 0;
    int ones = 0;
    for(int i=0; i < size; i++)
    {
        float* row = group[i];
        float class_label = row[cols-1];
        if(class_label == 0) zeroes++;
        if(class_label == 1) ones++;
    }
    if(ones >= zeroes)
    {
        return 1;
    }
    else {
        return 0;
    }
}

//
// recursive growth of a decision tree
//
void grow(struct Node* decision_tree, int max_depth, int min_samples_leaf, int max_features, int depth, int rows, int cols)
{
    struct Node obj = *decision_tree;
    struct var_array left_half = obj.two_halves[0];
    struct var_array right_half = obj.two_halves[1];

    float** left = left_half.array;
    float** right = right_half.array;
    decision_tree->two_halves = NULL;

    if(left == NULL || right == NULL)
    {
        float leaf = get_leaf_node_class(combine_arrays(left, right, left_half.length,right_half.length, cols), rows, cols);
        decision_tree->left_leaf = leaf;
        decision_tree->right_leaf = leaf;
        return;
    }
    if(depth >= max_depth)
    {
        decision_tree->left_leaf = get_leaf_node_class(left, left_half.length, cols);
        decision_tree->right_leaf = get_leaf_node_class(right, right_half.length, cols);
        return;
    }
    if(left_half.length <= min_samples_leaf)
    {
        decision_tree->left_leaf = get_leaf_node_class(left, left_half.length, cols);
    }
    else {
        struct split_params_struct split_params = calculate_best_data_split(left, max_features, left_half.length, cols);
        decision_tree->left = create_node();
        decision_tree->left->two_halves = split_params.two_halves;
        decision_tree->left->index = split_params.index;
        decision_tree->left->value = split_params.value;
        grow(decision_tree->left, max_depth, min_samples_leaf, max_features, depth+1, rows, cols);
    }
    if(right_half.length <= min_samples_leaf)
    {
        decision_tree->right_leaf = get_leaf_node_class(right, right_half.length, cols);
    }
    else {
        struct split_params_struct split_params = calculate_best_data_split(right, max_features, right_half.length, cols);
        decision_tree->right = create_node();
        decision_tree->right->two_halves = split_params.two_halves;
        decision_tree->right->index = split_params.index;
        decision_tree->right->value = split_params.value;
        grow(decision_tree->right, max_depth, min_samples_leaf, max_features, depth+1, rows, cols);
    }
}

//
// split a dataset into two parts for when calculating the gini index
// return an array of two halves, left and right
//
struct var_array* split_dataset(int index, float value, float** data, int rows, int cols)
{
    if(verbose) printf("\nsplitting dataset into two halves...\n\n");

    float** left = (float**) malloc(1 * sizeof(float)*cols);
    float** right = (float**) malloc(1 * sizeof(float)*cols);

    int left_count = 0;
    int right_count = 0;

    for(int i = 0; i < rows; i++)
    {
        // iterate single row now
        float* row = data[i];
        if(row[index] < value)
        {
            // copy the row into left half
            left[left_count] = row;
            left_count++;
            float** temp = realloc(left, left_count * sizeof(float) * cols);
            if (temp != NULL) left = temp;
        }
        else {
            // copy the row into right half
            right[right_count] = row;
            right_count++;
            float** temp = realloc(right, right_count * sizeof(float) * cols);
            if (temp != NULL) right = temp;
        }
    }
    struct var_array* left_right = malloc(sizeof(struct var_array) * 2);
    if(verbose) printf("\ntest split: %d | %d\n",left_count, right_count);

    left_right[0] = (struct var_array){left_count, left};
    left_right[1] = (struct var_array){right_count, right};
    return left_right;
}

//
// calculate the gini index for two halves split of data set
// class labels are either one or multiple class values:
// for binary classification - {0, 1}
//

float gini_index(struct var_array* two_halves, float* class_labels, int class_labels_count, int cols)
{
    if(verbose) printf("\ncalculating gini index based on split...\n");

    int count = 2; // two halves expected
    int n_instances = two_halves[0].length + two_halves[1].length;
    float gini = 0.0;
    for(int i=0; i < count; i++)
    {
        struct var_array group = two_halves[i];
        int size = group.length;
        if(size == 0) continue;
        // else compute the score
        float sum = 0.0;

        for(int j=0; j < class_labels_count; j++)
        {
            float class = class_labels[j];
            float occurences = 0;
            for(int k=0; k < size; k++)
            {
                float label = group.array[k][cols-1];
                if(label == class)
                {
                    occurences += 1.0;
                }
            }
            float p_class = occurences / (float)size;
            sum += (p_class * p_class);
        }
        gini += (1.0 - sum) * ((float)size / (float)n_instances);
    }

    if(verbose) printf("\ngini: %f \n", gini);
    if(verbose) printf("-----------------------------------------\n");
    return gini;
}

//
// calculate the best splitting for the data set
//
struct split_params_struct calculate_best_data_split(float** data, int max_features, int rows, int cols)
{
    if(verbose) printf("\ncalculating best split for dataset...\n");
    if(verbose) printf("\nrows: %d\ncols: %d\n",rows,cols);

    struct class_label_struct classes_struct = get_class_values(data, rows, cols);
    struct var_array* best_two_halves = NULL;

    float best_value = (float)INT_MAX;
    float best_gini = (float)INT_MAX;
    int best_index = INT_MAX;

    // create a features array and initialize to avoid non-set memory
    int* features = malloc(max_features* sizeof(int));
    for(int i=0; i < max_features; i++) features[i] = -1;

    int count = 0;
    while(count < max_features)
    {
        int max = cols-2;
        int min = 0;
        int index = rand() % (max + 1 - min) + min;
        if(!contains_int(features, max_features, index))
        {
            if(verbose) printf("\nadding unique index: %d\n",index);
            features[count] = index;
            count++;
        }
    }
    if(verbose) printf("\n-----------------------------------------\n");
    for(int i=0; i < max_features; i++)
    {
        int index = features[i];
        for(int j=0; j < rows; j++)
        {
            float* row = data[j];
            struct var_array* two_halves = split_dataset(index, row[index], data, rows, cols);
            float gini = gini_index(two_halves, classes_struct.class_labels, classes_struct.count, cols);

            if(gini < best_gini)
            {
                best_index = index;
                best_value = row[index];
                best_gini = gini;
                best_two_halves = two_halves;
            }
        }
    }
    // free things out
    free(features);
    return (struct split_params_struct){best_index, best_value, best_gini, best_two_halves};
}

//
// returns a random sample of the data set that is the fraction ratio of the size of dataset
//
float** subsample(float** data, float ratio, int rows, int cols)
{
    int sample_rows = (int)((float)rows * ratio);
    float** sample = (float**) malloc(sample_rows * sizeof(float) * cols);

    int* indecies = malloc(sample_rows * sizeof(int));
    int count = 0;
    for(int i=0; i < sample_rows; i++) indecies[i] = -1;

    while(count < sample_rows)
    {
        int max = rows-1;
        int random_index = rand() % (max + 1);
        if(!contains_int(indecies, sample_rows, random_index))
        {
            if(verbose) printf("\nrandom index:%d",random_index);
            sample[count] = data[random_index];

            // keep track of this index so that there are no duplicate rows
            indecies[count] = random_index;
            count++;
        } else {
            if(verbose) printf("\n duplicate encountered in sub sampling\n");
        }
    }
    if(verbose) printf("\nreturning a subsample of size: %dx%d\n",sample_rows,cols);
    return sample;
}

//
// predict by propagating the trained tree
//
float predict(struct Node *decision_tree, float* row)
{
    if(row[decision_tree->index] < decision_tree->value)
    {
        if(decision_tree->left != NULL)
        {
            return predict(decision_tree->left, row);
        }
        else
        {
            return decision_tree->left_leaf;
        }
    } else {
        if(decision_tree->right != NULL)
        {
            return predict(decision_tree->right, row);
        }
        else
        {
            return decision_tree->right_leaf;
        }
    }
}

//
// determines the prediction based on a majority vote from an array of trees (forest)
// currently only implemented for binary classification -> {0,1}
//
float majority_vote_predict(struct Node** trees, int n_estimators, float* row)
{
    int zeroes = 0;
    int ones = 0;
    for(int i=0; i < n_estimators; i++)
    {
        float prediction = predict(trees[i], row);
        if(verbose) printf("\nsingle prediction: %f",prediction);
        if(prediction == 0) zeroes++;
        if(prediction == 1) ones++;
    }
    if(ones > zeroes)
    {
        if(verbose) printf("\nmajority vote: 1\n");
        return 1;
    }
    else {
        if(verbose) printf("\nmajority vote: 0\n");
        return 0;
    }
}

//
// get predictions for the entire model of trees (random forest)
//
float* get_predictions(float** test_data, int rows_test, struct Node** trees, int n_estimators)
{
    // predictions for each row
    // count of rows: rows_test
    float* predictions = malloc(rows_test * sizeof(float));
    for(int i=0; i < rows_test; i++)
    {
        float* row = test_data[i];
        float prediction = majority_vote_predict(trees, n_estimators, row);
        predictions[i] = prediction;
    }
    return predictions;
}