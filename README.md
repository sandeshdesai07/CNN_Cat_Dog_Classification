[README_Cat_Dog.md](https://github.com/user-attachments/files/28864111/README_Cat_Dog.md)
# CNN Cat vs Dog Classification

A **Convolutional Neural Network (CNN)** project built with **TensorFlow/Keras** to classify an input image as either **cat** or **dog**.

## Project Overview

This notebook trains a binary image classification model on the **Dogs vs Cats** dataset.  
The workflow includes:

- loading images from directory-based datasets,
- preprocessing and normalization,
- building a CNN model,
- training and validation,
- saving the trained model,
- testing predictions on sample images.

## Dataset

The notebook uses the Kaggle dataset:

**`salader/dogsvscats`**

The data is loaded into:

- `train/` for training
- `test/` for validation

Images are resized to **256 × 256** and processed in batches of **32**.

## Model Architecture

The CNN model is built using the following layers:

- **Conv2D**
- **BatchNormalization**
- **MaxPooling2D**
- **Flatten**
- **Dense**
- **Dropout**
- Final **Sigmoid** output layer for binary classification

### Training Details

- **Loss Function:** `binary_crossentropy`
- **Optimizer:** `AdamW`
- **Learning Rate:** `1e-4`
- **Weight Decay:** `1e-4`
- **Epochs:** `5`
- **Batch Size:** `32`

## Preprocessing

All images are normalized before training:

```python
image = tf.cast(image / 255.0, tf.float32)
```

This scales pixel values to the range **0 to 1**, which helps the model train more effectively.

## Model Saving

After training, the model is saved in both formats:

- **HDF5 format:** `cat-vs-dogs-trained-modell.h5`
- **Keras native format:** `cat-vs-dogs-trained-modell.keras`

## Results

During training, the model achieved improving accuracy across epochs and was tested on example cat/dog images for prediction.

## Download Pretrained Weights

To avoid training the model from scratch and to get proper output immediately, download the pretrained weights from the links below:

- [Download Weights 1](https://drive.google.com/file/d/1BqMk1hTao_1Hoxs2ftpN5QrI4TA7BXr3/view?usp=sharing)
- [Download Weights 2](https://drive.google.com/file/d/1y9iyszI0W-BrfomSfGPZH69eMyqvDSbu/view?usp=sharing)

## How to Run

1. Download the dataset.
2. Place the training and testing images in the correct folders.
3. Install the required libraries:
   ```bash
   pip install tensorflow matplotlib opencv-python kaggle
   ```
4. Load the pretrained weights or train the model from scratch.
5. Run inference on your own images.

## Example Prediction Flow

1. Read an image using **OpenCV**.
2. Resize it to **256 × 256**.
3. Reshape it to match model input:
   ```python
   image = image.reshape((1, 256, 256, 3))
   ```
4. Use `model.predict()` to get the output.
5. Interpret:
   - output close to **0** → one class
   - output close to **1** → the other class

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV

## Notes

- The notebook is designed for binary classification only.
- Ensure the image paths are correct before running inference.
- The model expects input images in **RGB format** with size **256 × 256**.

## Acknowledgement

Dataset used from Kaggle: **Dogs vs Cats** by `salader`.
