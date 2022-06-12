#ifndef argparse_h
#define argparse_h

#include <stdlib.h>
#include <argp.h>

/* How many arguments we accept. */
#define COUNT_ARGS 1

const char *argp_program_version =
    "random-forests-c 1.0";
const char *argp_program_bug_address =
    "https://github.com/dobroshynski/random-forests-c";

/* Program documentation. */
static char doc[] =
    "random-forests-c -- Basic implementation of random forests and accompanying decision trees in C";

/* A description of the arguments we accept. */
static char args_doc[] = "CSV_FILE";

/* The options we understand. */
static struct argp_option options[] = {
    {"num_rows", 'r', "number", 0, "Optional number of rows in the input CSV_FILE, if known", 0},
    {"num_cols", 'c', "number", 0, "Optional number of cols in the input CSV_FILE, if known", 0},
    {"log_level", 'l', "number", 0, "Optional debug logging level [0-3]. Level 0 is no output, 3 is most verbose. Defaults to 1.", 1},
    {"seed", 's', "number", 0, "Optional random number seed.", 2},
    {0}};

/* Used by main to communicate with parse_opt. */
struct arguments
{
    char *args[COUNT_ARGS]; /* CSV file argument. */

    long rows, cols;
    int log_level;
    int random_seed;
};

/* Parse a single option. */
static error_t
parse_opt(int key, char *arg, struct argp_state *state)
{
    /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
    struct arguments *arguments = state->input;

    switch (key)
    {
    case 'r':
        arguments->rows = atol(arg);
        break;
    case 'c':
        arguments->cols = atol(arg);
        break;
    case 'l':
        arguments->log_level = atoi(arg);
        break;
    case 's':
        arguments->random_seed = atoi(arg);
        break;

    case ARGP_KEY_ARG:
        if (state->arg_num >= COUNT_ARGS)
            /* Too many arguments. */
            argp_usage(state);

        arguments->args[state->arg_num] = arg;

        break;

    case ARGP_KEY_END:
        if (state->arg_num < COUNT_ARGS)
            /* Not enough arguments. */
            argp_usage(state);
        break;

    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

#endif // argparse_h
