# Cifar Classification - Homework 4 - Evan Kluger

## Summary
Here is my work for Homework 4 - The Cifar Dataset. I had experimented a lot to find the right parameters for the best results, but ultimately due to changing my architecture at the last minute before needing to hand this in, I only had a couple runs before kicking off my test run. I had a target of 90% accuracy for the Cifar-10 Dataset, which I fell short off by only acheiving an 85.70% accuracy. On the Cifar100 dataset I also fell short and achieved an 80.27% accuracy.

## Results
- **CIFAR-10 Test Accuracy:** 85.70%
- **CIFAR-100 Top 5 Test Accuracy:** 80.27%

## Model Architecture

1. **Conv2D:** I used my Conv2D class from the previous assignment which includes a He initilization of the weights

2. **Classifier:** The Classifier class contains architecture to the advice that I recieved during Office Hours in the following structure -> 
    - Initialized with a Conv.
    - Added 2 ResidualBlocks.
    - Used max pooling to reduce from 32x32 to 16x16 dimensions.
    - Added 2 ResidualBlocks.
    - Another max pooling to further scale down to 8x8.
    - Another 2 ResidualBlocks
    - More pooling to further scale down to 1x1.
    - 1x1 is passed through Fully Connected Layer

3. **GroupNorm:** I adapted the main literature for GroupNorm to fit for this project

4. **ResidualBlock:** I made the ResidualBlock composed of 3x Conv -> Group Norm -> Relu with Skip Connection.

## Data Augmentation

I used the following 9 Data Augmentations.

1. **Brightness**
2. **Contrast**
3. **Saturation**
4. **Hue**
5. **Cutout:** 
6. **Random cropping**
7. **Horizontal and Vertical Flips**
8. **Rotation**
9. **Noise addition**

When I first was playing around with the data augmentation, I had the probabilities formatted in a way that certain batches would recieve data augmentations X percent of the time. For example, the brightness data augmentation would be applied to a batch 25% of the time. After thinking about it a bit, I felt that instead of X percent of batches recieving certain transformations that instead X percent of images should recieve the transformations, that way certain batches contain a mix of different augmentations on different images. To do this I performed a for loop over all the images in a batch and per image basis applied a transformtaion based on the percentage. This worked better than before but ended up costing a lot of extra time to run the program. Instead I rearranged my data augmentation fucntion to not do a for loop over the whole batch but still retain the per image basis. To do this for each transform I created probability masks that are filled with values between 0-1 with the same shape as the batch. Then the probability mask values are set to True/False based on if they are greater/less than the probability for the data augmentation to occur. Next the data augmentation is applied to all images where the probability mask is set to True and the image tensor is updated.

The choice of which of the nine data augmentations have higher/lower probailities was decided based on a combintaion of test and research. I found online that data augmentations such as Crop, Cutout, and Horizontal flips lead to better performance for Cifar and I validated this with my own testing. This testing was done before I fixed my architecture so the testing helped my current model at the time go from around 50% -> 63%, and although I ran out of time to retest the best data architecture with my new architecture I assumed the same findings from testing the data augmentations would lead to better results.

## Experimentation

During experimentation I found that increasing the layer depths of my model and adjusting the data augmentations led to better results. After fixing up my architecture, I had only one try left before it was time to hand in so for my final validation test I did the following. I chose a layer depths of [32, 64, 128, 256, 512] with 3 by 3 kernel sizes and continued raising the batch size until I recieved a gpu error. I was able to get my batch size to be about 1300, and I set the num iterations to 3000 and kicked it off. After the validation results looked solid, I kicked it off again this time at 1500 batch size and for 5000 iterations and got my test results.

## Conclusion

I did not meet the state of the art goal I set for myself for the Cifar 10 dataset of 90% accuracy, but I am satisfied with getting close with around 85%. With more time to play around and tweek my model I believe I can get there. For the Cifar100 dataset I believe simiraly that although I did not meet the required top 5 90% accuracty, I am happy with the progress I have made. 

## Thanks, Evan Kluger