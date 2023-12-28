# TransformerBlock - Homework 6 - Evan Kluger

## Overview
This code implements the TransformerBlock class and MultiHeadAttention class based on the Transformer papers from class. 

## Files Description

### `transformer.py`
Contains the core implementation of the Transformer model with the `MultiHeadAttention` and `TransformerBlock` classes. 

### `transformer_output_tests.py`
Includes tests for output shapes and general functionality of the Transformer classes. This file ensures that the Transformer model and its components produce outputs with expected dimensions and types.

### `transformer_man_bites_dog.py`
Demonstrates an application of the Transformer model where it is trained to predict the next word in the sequence "man bites dog."

### `transformer_reverse_seq_test.py`
Contains a test case where the Transformer model is trained to predict the reverse of a sequence of integers. 

### `transformer_attention_test.py`
Focuses on testing the attention mechanism of the Transformer model

### `transformer_mask_test.py`
Tests the masking functionality of the Transformer model, particularly the causal masking required for parallel training in sequence modeling. This file uses Jacobian matrices to verify the correctness of the masking.

### `common_functions.py`
Provides common utility functions used in various test files.

### `Plots`
There are a few plot that are generated after running the two of the test files. These plots will indicate the loss as a function of iteration to demonstarte the learning of the Transformer as it undergoes training for man bites dog and reverse sequence tests.


### `Results`
Man Bites Dog -> Correclty predicts 'dog' after bites.

Reverse Sequence -> Returns a test accuracy of 49% after the training with the current parameters in the file. 