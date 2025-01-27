# Traffic Sign Classifier using Transfer Learning with ResNet18

This project leverages transfer learning using the ResNet18 model along with a custom fully connected (FC) layer to classify traffic signs into 43 different classes. The model is trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset, which is publicly available on [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Contact](#contact)

## Project Overview

This project aims to create a traffic sign classification system using a convolutional neural network (CNN). We utilize the ResNet18 model, a pre-trained model, as a feature extractor for transfer learning. We add a custom fully connected layer to this architecture to make predictions across the 43 classes of traffic signs present in the GTSRB dataset.

The GTSRB dataset is preprocessed, and the ResNet18 model is fine-tuned to adapt to the traffic sign classification task.

## Dataset

The dataset used in this project is the [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset, which contains labeled images of 43 different types of traffic signs. The dataset consists of:

- **Train**: Training images organized into subfolders, each representing a different traffic sign class.
- **Test**: Test images organized in a similar fashion.

### Dataset Structure

The dataset should be organized in the following structure:

GTSRB/ │ ├── train/ │ ├── class_0/ │ ├── class_1/ │ ├── ... │ └── class_42/ │ ├── test/ │ ├── class_0/ │ ├── class_1/ │ ├── ... │ └── class_42/

## Installation

To get started, you need to install the necessary dependencies. You can do so by using the following command:

```bash
pip install -r requirements.txt
```
The `requirements.txt` file includes the necessary libraries such as:

- `torch`
- `torchvision`
- `matplotlib`
- `pandas`
- `numpy`
- `scikit-learn`
- `opencv-python`
- `PIL`

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/abhisekh2004/traffic-sign-classifier.git
    cd traffic-sign-classifier
    ```

2. Download the GTSRB dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and unzip it into the `GTSRB` directory.

3. Run the training script:

    ```bash
    python traffic_sign_classifier.py
    ```

4. After training, the model will be saved in the `models` directory. You can load and evaluate the model with:

    ```bash
    python model_test.py --model_path models/best_model.pth
    ```
    
## Model Architecture

The model uses ResNet18 as the base architecture, pre-trained on the ImageNet dataset. A custom fully connected (FC) layer is added to the end of the model to classify the traffic signs into 43 different classes. The architecture is as follows:

- **ResNet18**: Pre-trained convolutional layers to extract features.
- **Fully Connected Layer**: Custom layer to classify the extracted features into 43 classes.

### Model Summary:

- **Input Size**: 224x224 (Image size).
- **Output Classes**: 43 (corresponding to 43 traffic sign categories).
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam Optimizer.

## Training

- **Epochs**: The model is trained for a fixed number of epochs (default: 20).
- **Batch Size**: The batch size can be adjusted depending on system memory.
- **Learning Rate**: The learning rate starts at 0.001 and decays throughout the training.

During training, the model's accuracy and loss are printed at each epoch, and the best-performing model is saved.

### Training Log Example:

```yaml
Epoch [1/20], Loss: 1.24, Accuracy: 75.2%
Epoch [2/20], Loss: 0.96, Accuracy: 81.6%
...
Epoch [20/20], Loss: 0.18, Accuracy: 97.8%
```
### Contact
Feel free to contribute by submitting issues or pull requests. If you have any questions, feel free to reach out to me at [abhisekh2004](https://github.com/abhisekh2004).

