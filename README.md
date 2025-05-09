# 🖼️ Advanced CNN Autoencoder for MNIST Image Denoising

This project implements an advanced Convolutional Neural Network (CNN)-based autoencoder to remove noise from handwritten digit images. Trained on the MNIST dataset, the model utilizes a U-Net inspired architecture and incorporates modern deep learning techniques for improved denoising performance.

## 📖 About the Project

The goal of this project is to demonstrate the application of deep learning, specifically autoencoders and CNNs, to the task of image denoising. By training a model to reconstruct clean images from their noisy counterparts, we can effectively reduce the impact of noise. The project uses the MNIST dataset as a case study and builds a robust autoencoder architecture incorporating features like learned upsampling and batch normalization to achieve better results.

## ✨ Features

* **CNN Autoencoder:** A powerful deep learning model designed for image-to-image tasks like denoising.
* **U-Net Inspired Architecture:** Employs a structure with skip connections to enhance the reconstruction of image details.
* **MNIST Dataset:** Uses the standard MNIST dataset, corrupted with Gaussian noise, for training and evaluation.
* **Advanced Techniques:** Incorporates `Conv2DTranspose` for learned upsampling and `BatchNormalization` for stable training.
* **Effective Training:** Utilizes `EarlyStopping` and `ReduceLROnPlateau` callbacks to optimize the training process and prevent overfitting.
* **Quantitative Evaluation:** Assesses denoising performance using standard metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
* **Visual Results:** Includes code to visualize the original, noisy, and denoised images for qualitative assessment.

## 🚀 Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

* Python 3.6+
* Jupyter Notebook (or JupyterLab)
* Required Python libraries (listed in Requirements)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/indranil143/Image_Denoiser.git
    cd Image_Denoiser
    ```
2.  Install the required libraries. It's recommended to use a virtual environment:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Advanced CNN Autoencoder for MNIST Image Denoising.ipynb"
    ```
2.  Run all the cells in the notebook sequentially.
3.  The notebook will perform the following steps:
    * Load and preprocess the MNIST data.
    * Add Gaussian noise to the images.
    * Define, compile, and train the CNN autoencoder model.
    * Evaluate the trained model using PSNR and SSIM.
    * Display example images showing the original, noisy, and denoised versions.

## 🏗️ Model Architecture

The core of the project is a CNN-based autoencoder with a structure similar to a U-Net.

* **Encoder:** Consists of convolutional layers followed by Batch Normalization and MaxPooling layers to downsample the input image and extract features.
* **Decoder:** Composed of `Conv2DTranspose` layers for learned upsampling, followed by Batch Normalization and convolutional layers. Skip connections concatenate feature maps from the encoder to corresponding decoder layers, helping preserve spatial information.
* **Output Layer:** A final convolutional layer with sigmoid activation outputs the denoised image with pixel values in the \[0, 1] range.

## 📊 Model Performance

The trained autoencoder achieved the following results on the noisy MNIST test dataset:

* **Average PSNR:** 20.96 dB
* **Average SSIM:** 0.8371

These metrics indicate the model's effectiveness in reconstructing the original image quality and structural details after denoising.

## 📁 Project Structure

* `Advanced CNN Autoencoder for MNIST Image Denoising.ipynb`: The main Jupyter Notebook containing all the code for data handling, model building, training, evaluation, and visualization.
* `Image Denoising Using CNN-Based Autoencoders.ipynb`: This notebook contains the code for the **previous, simpler CNN autoencoder model**.
* `requirements.txt`: Lists the Python libraries required to run the project.
* `LICENSE`: File containing the project's license information.

## 🔬 Comparison with Previous Project

This project is an evolution of a previous attempt at image denoising using a simpler CNN autoencoder architecture. Here's a summary of the previous project's details and a comparison of the results:

### Previous Project: Simple CNN Autoencoder

* **Objective:** Remove noise from MNIST images using a basic CNN-based Autoencoder.
* **Dataset:** MNIST – 70,000 grayscale digit images (28x28 pixels), also with Gaussian noise.
* **Model:**
    * Encoder: Simpler architecture with Conv2D layers and MaxPooling.
    * Decoder: Used Conv2D layers with simple UpSampling.
* **Training:**
    * Optimizer: Adam
    * Loss Function: Binary cross-entropy
    * Batch Size: 128
    * Epochs: 10
* **Results:**
    * Final Test Loss: 0.0974
    * Average PSNR: 19.75 dB
    * Average SSIM: 0.8637

### Conclusion on Comparison

Comparing the two projects, the **Advanced CNN Autoencoder for MNIST Image Denoising (U-Net Inspired)** demonstrates improved denoising performance, primarily evidenced by a higher **Average PSNR (20.96 dB vs 19.75 dB)**. PSNR is a key metric for evaluating noise reduction, and the increase indicates that the advanced model is more effective at minimizing pixel-wise errors and restoring image quality. While the previous project achieved a slightly higher Average SSIM, the significant gain in PSNR in the advanced model suggests it provides a better overall balance of noise removal and detail preservation for this task. The architectural improvements, including `Conv2DTranspose`, `BatchNormalization`, and skip connections (U-Net structure), along with the use of MSE loss, contributed to this enhanced performance.


## ✅ Requirements

The project requires the following Python libraries:

* `numpy`
* `tensorflow`
* `matplotlib`
* `scikit-image` (for PSNR and SSIM metrics)


### Example Outputs:
The following images show the comparison between the original, noisy, and denoised images for a sample test:
![Sample Image](https://github.com/indranil143/Image_Denoiser/blob/main/SS.png)

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please feel free to fork the repository, create a new branch, make your changes, and open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* The MNIST dataset for providing the training data.
* The TensorFlow and Keras teams for providing the deep learning framework.
* The scikit-image library for image quality metrics.
* The developers of other open-source libraries used in this project.

---
© 2025 indranil143
