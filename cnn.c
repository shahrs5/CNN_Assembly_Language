#include <stdio.h>
#include <math.h>

#define INPUT_SIZE 28
#define FILTER_SIZE 5
#define CONV_OUTPUT_SIZE 24
#define POOL_SIZE 2
#define POOL_OUTPUT_SIZE 12
#define NUM_FILTERS 8
#define DENSE_INPUT_SIZE (POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE * NUM_FILTERS)
#define NUM_CLASSES 10

// Input, filters, biases, and outputs
float input[INPUT_SIZE][INPUT_SIZE];
float filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE];
float biases[NUM_FILTERS];
float output_filter[NUM_FILTERS][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE];
float output_pool[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE];
float flattened_input[DENSE_INPUT_SIZE];
float dense_weights[NUM_CLASSES][DENSE_INPUT_SIZE];
float dense_biases[NUM_CLASSES];
float dense_output[NUM_CLASSES];
float softmax_output[NUM_CLASSES];

// Convolution
void conv2d(float input[INPUT_SIZE][INPUT_SIZE], float filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE], float biases[NUM_FILTERS], float output[NUM_FILTERS][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])
{
    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < CONV_OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < CONV_OUTPUT_SIZE; j++)
            {
                float sum = 0.0f;
                for (int ki = 0; ki < FILTER_SIZE; ki++)
                {
                    for (int kj = 0; kj < FILTER_SIZE; kj++)
                    {
                        sum += input[i + ki][j + kj] * filters[f][ki][kj];
                    }
                }
                output[f][i][j] = sum + biases[f];
            }
        }
    }
}

// Max pooling
void maxpool(float input[NUM_FILTERS][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float output[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE])
{
    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < POOL_OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < POOL_OUTPUT_SIZE; j++)
            {
                float max_val = input[f][i * 2][j * 2];
                for (int ki = 0; ki < 2; ki++)
                {
                    for (int kj = 0; kj < 2; kj++)
                    {
                        float val = input[f][i * 2 + ki][j * 2 + kj];
                        if (val > max_val)
                            max_val = val;
                    }
                }
                output[f][i][j] = max_val;
            }
        }
    }
}

// Flatten
void flatten(float input[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float output[DENSE_INPUT_SIZE])
{
    int index = 0;
    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < POOL_OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < POOL_OUTPUT_SIZE; j++)
            {
                output[index++] = input[f][i][j];
            }
        }
    }
}

// Dense layer
void denselayer(float input[DENSE_INPUT_SIZE], float weights[NUM_CLASSES][DENSE_INPUT_SIZE], float biases[NUM_CLASSES], float output[NUM_CLASSES])
{
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < DENSE_INPUT_SIZE; j++)
        {
            sum += input[j] * weights[i][j];
        }
        output[i] = sum + biases[i];
    }
}

// Softmax with Taylor approximation
float exp_taylor(float x)
{
    float term = 1.0f;
    float sum = 1.0f;
    for (int i = 1; i <= 10; i++)
    {
        term *= x / i;
        sum += term;
    }
    return sum;
}

void softmax(float input[NUM_CLASSES], float output[NUM_CLASSES])
{
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        output[i] = exp_taylor(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        output[i] /= sum;
    }
}

// Print functions
void print_matrix(const char *name, float matrix[INPUT_SIZE][INPUT_SIZE])
{
    printf("%s:\n", name);
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            printf("%.1f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_probs(float probs[NUM_CLASSES])
{
    printf("\nClass Probabilities:\n");
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        printf("Class %d: %.4f\n", i, probs[i]);
    }
}

int main()
{
    // Dummy initialization
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            input[i][j] = (i + j) % 5;

    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < FILTER_SIZE; i++)
            for (int j = 0; j < FILTER_SIZE; j++)
                filters[f][i][j] = 0.05f;
        biases[f] = 0.1f;
    }

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < DENSE_INPUT_SIZE; j++)
            dense_weights[i][j] = 0.01f;
        dense_biases[i] = 0.1f;
    }

    conv2d(input, filters, biases, output_filter);
    maxpool(output_filter, output_pool);
    flatten(output_pool, flattened_input);
    denselayer(flattened_input, dense_weights, dense_biases, dense_output);
    softmax(dense_output, softmax_output);

    print_probs(softmax_output);
    return 0;
}
