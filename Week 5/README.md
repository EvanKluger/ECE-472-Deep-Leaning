# AG News Text Classifier
# AG News Classification - Homework 5 - Evan Kluger

## Model
I used the `MiniLM` model variant for the text classification task. MiniLM is a version of the BERT model

## Tokenizer
I employed the `sentence-transformers/all-MiniLM-L6-v2` tokenizer from the `transformers` library

## Data Acquisition
The `ag_news` dataset was  imported using the `datasets` library

## Execution Environment
All computations and training were performed on Google Colab where I was able to run my code on an A100 GPU -> up to 40 GB of GPU space.

## Results
The final test accuracy is 91.18%.

## Note
I had an issue with calculating the final test accuracy with initilizing such a big tensor. I ended up "batching" the final test cases and just going through them and classifying each case in the batch and then adding the results ot the total correct. Next, I took all the corrects and divided by total cases to get final test accuracy.