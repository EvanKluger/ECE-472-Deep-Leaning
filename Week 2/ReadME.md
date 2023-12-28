To Run this Program Download all the files and:


$ python3 -m venv venv
$ source ./venv/bin/activate
(venv) $ pip install tensorflow matplotlib pyyaml tqdm pytest
(venv) $ python mlp_spiral.py  
(venv) $ python -m pytest mlp_spiral_tests.py




Note:

To create my function f I tested the different parameters individually to find the resulting best plot. 
I started off with using the Relu for the hidden layer activation fucntion and the Sigmoid for the outlayer
activation function. I chose these because these were the activation functions discussed in class. I did not exchange these functions for any others because while testing it was the parameters that made the substantial difference not the activation fucntions. For the parameters num_hidden_layers and num_layer_widths I found that increasing the num_layer_widths while maintaining a smaller num_hidden_layers
resulted in best response. After finding this out, I tested different values along these rnages and came to my final numbers. 
Lastly for the L2 constant, I started off with a high value and found that decreasing it substantially led to accurate results.