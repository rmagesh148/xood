# Confidence Prediction For Deep Image Classification

### Algorithm summary
1. Start with a fully trained classification model.
2. Extract features around relu layers.
3. Split features to improve linear separation.
4. Fit logistic regression on distorted dataset to predict confidence from the features.

![alt text](https://github.com/rmagesh148/xood/blob/main/min_max_2.jpg?raw=true)

Distribution of maximum values before an activation layer inside ResNet34. Green represents CIFAR-10 (ID) and  Red represents LSUN-Resize(OOD) (left figure) or SVHN(OOD) (right figure). Note how in-distribution data lies in a single mode, while out-of-distribution data can be on either side.

### Project Structure
The confidence algorithm is in confidenciator.py.

### Dependencies
* Python 3.7 or 3.8.
* Numpy sklearn, matplotlib, scipy, skimage, pytorch, torchvision.

### Set up conda environment
conda create -n env_name python=3.8 pytorch numpy matplotlib scipy scikit-learn pandas scikit-image tensorflow-gpu torchvision cudatoolkit=11.3 -c pytorch
