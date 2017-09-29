#include "cnn/layer.h"

Layer::Layer(int input_rows, int input_cols, int output_rows, int output_cols) :
    input_(input_rows, input_cols),
    output_(output_rows, output_cols),
    input_gradients_(input_rows, input_cols) {}
