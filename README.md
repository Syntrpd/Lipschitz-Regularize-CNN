# Lipschitz-Regularize-CNN

Lipschitz Regularize deep Convolutional Neural Networks (CNNs) with PyTorch

## Model Architecture

1. DnCNN

![](images/dncnn.png)

## Usage 

1. Import Image-Denoise/src/ADMMProjectLayer.py and Image-Denoise/src/Utils.py.
2. ```python
   from ADMMProjectLayer import LipConstrainLayer
   ```
3. Use the function LipConstrainLayer that takes the conv layer, image size and Lipschitz Constant as input
 
## Dataset

Images from [Berkeley Segmentation Dataset and Benchmark](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).  

* Download here: [https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz)

It contains two sub-directories: `train` and `test`, which consist of 200 and 100 images, respectively, of either size `321 × 481` or `481 × 321`. While we saw that thousand to millions of images were required for image classification, we can use a much smaller training set for image denoising. This is because denoising each pixel of an image can be seen as one regression problem. Hence, our training is in fact composed of `200 × 321 × 481` **≈ 31 million** samples.

## Demo Usage

1. Clone this repository

```bash
git clone https://github.com/Syntrpd/Lipschitz-Regularize-CNN.git
```

2. Download dataset

3. Train the model

```bash
python Image-Denoise/src/main.py
```

**PS**: Read [`argument.py`](Image-Denoise/src/argument.py) to see what parameters that you can change.  

## Demonstration and tutorial

Nil

## References
Nil
