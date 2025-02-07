# Image-Denoising-Using-CNN-Based-Autoencoders

This project implements an image denoising technique using Convolutional Autoencoders (CNN-based) to remove noise from images, with a focus on improving image quality in noisy datasets.

## Overview

In this project, a Convolutional Neural Network (CNN) based Autoencoder is trained to denoise images from the MNIST dataset. The goal is to learn an efficient mapping between noisy and clean images by using an unsupervised learning approach. The model leverages the power of CNNs for feature extraction and reconstruction, combined with the architecture of autoencoders to restore the original clean images.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Structure

- ├── README.md
- ├── denoising_model.py
- ├── data_preprocessing.py
- ├── requirements.txt
- └── results/

## Installation

### Prerequisites

Make sure you have Python 3.6 or higher installed, as well as the following libraries:

- TensorFlow
- NumPy
- Matplotlib

You can install the required libraries using `pip`:
```
pip install -r requirements.txt
```
Alternatively, you can manually install the dependencies using:
```
pip install tensorflow numpy matplotlib
```
Clone the repository:
```
git clone https://github.com/indranil143/Image-Denoising-Using-CNN-Based-Autoencoders.git
cd Image-Denoising-Using-CNN-Based-Autoencoders
```
Run the model:
Execute the denoising_model.py script to train the autoencoder:
```
python denoising_model.py
```
## usage
This will:
- Load the MNIST dataset.
- Add noise to the images.
- Train the convolutional autoencoder model.
- Save the trained model and output images in the results/ folder.
  
Visualize Results:
After training, the denoised images will be visualized, comparing the original, noisy, and denoised images. The results will be displayed directly on your screen.

## Results
### Performance Metrics:
- Final Test Loss: 0.0974
- Average PSNR: 19.75 dB
- Average SSIM: 0.8637
  
These values indicate the effectiveness of the model in denoising the images and retaining structural details.

### Example Outputs:
The following images show the comparison between the original, noisy, and denoised images for a sample test:
![Sample Image](https://github.com/indranil143/Image-Denoising-Using-CNN-Based-Autoencoders/blob/main/sample%20image.png)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
