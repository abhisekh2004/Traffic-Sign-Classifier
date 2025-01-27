Traffic Sign Classifier using Transfer Learning (ResNet18)
Classify traffic signs into 43 categories using the GTSRB dataset and transfer learning with ResNet18.

🌟 Overview
Traffic sign classification plays a crucial role in autonomous driving systems. This project leverages ResNet18 pretrained on ImageNet and fine-tuned with a custom fully connected layer to classify 43 different traffic signs. The model is trained and tested using the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

📂 Dataset
The GTSRB dataset contains labeled images of traffic signs divided into 43 classes.
Dataset link: GTSRB Dataset on Kaggle

Folder Structure
train/: Training images organized into 43 subfolders (one for each class, class_0 to class_42).
test/: Test images for evaluating the model.
🧠 Model Architecture
Base Model: ResNet18 (pretrained on ImageNet).
Custom Fully Connected (FC) Layer:
Input: 512 features (from ResNet18 output).
Hidden Layer: 256 neurons with ReLU activation.
Output Layer: 43 neurons (one for each traffic sign class).
The ResNet18 model extracts general image features, while the custom FC layer adapts the model for traffic sign classification.

⚙️ Installation
Clone this repository:

bash
Copy
Edit
git clone https://github.com/abhisekh2004/Traffic-Sign-Classifier.git
cd Traffic-Sign-Classifier
Install required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset:

Visit the GTSRB Dataset.
Extract the dataset into the project directory, ensuring it contains train/ and test/ folders.
🚀 Training the Model
Train the model using the training dataset:

bash
Copy
Edit
python train.py
The trained weights will be saved as resnet_custom_fc.pth.

📊 Results
The model successfully classifies traffic signs into 43 categories.
Loss Graph: Training and testing losses converge over epochs.
Accuracy: Achieves strong performance on the test dataset.
📖 References
GTSRB Dataset: Kaggle Link
ResNet18: Deep Residual Learning for Image Recognition
PyTorch Framework: PyTorch Official Documentation
🌐 About Me
Developed by abhisekh2004. Contributions and suggestions are welcome!
