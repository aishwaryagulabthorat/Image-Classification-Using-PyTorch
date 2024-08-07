# Image-Classification-Using-PyTorch
The project focuses on classifying images of playing cards using a deep learning model. By utilizing EfficientNet-B0 and transfer learning, built a model that can classify playing card images with high accuracy. 


**Objective**: The project focuses on classifying images of playing cards using a deep learning model.

**Dataset**: The dataset consists of images of playing cards, divided into training, validation, and test sets.

**Key Components**

**1.	Dataset Preparation:**
o	PlayingCardDataset Class: A custom dataset class leveraging ImageFolder from torchvision to load and preprocess images. The dataset is transformed to resize images to 128x128 pixels and convert them to tensors.
o	DataLoader: Utilized to load data in batches, which helps in efficient training and validation.

**2.	Model Architecture:**
o	Base Model: EfficientNet-B0, known for its state-of-the-art performance with fewer parameters compared to other models. EfficientNet-B0 is used to extract features from the images.

o	Pre-trained Weights: The model uses pre-trained weights from efficientnet_b0_ra-3dd342df.pth to leverage transfer learning, which helps in speeding up the training process and improving accuracy.

o	Classifier: 

Custom Model:

•	Feature Layers: In the custom model SimpleCardClassifer, the self.features attribute contains the convolutional layers of EfficientNet-B0 (excluding the final classification layer). This segment of the model performs the core convolutional operations. Here we basically remove the last layer of final classification, because we need to do that classification on our data. This step will just extract features from the pre-trained model EfficientNet-B0.

•	Final Classifier: After feature extraction, the flattened output is passed to a fully connected layer (the classifier), which outputs the final class predictions.

**3.	Training and Validation:**
o	Loss Function: CrossEntropyLoss, commonly used for classification tasks.

o	Optimizer: Adam optimizer with a learning rate of 0.001, chosen for its efficiency in handling sparse gradients.

o	Training Loop: Iterates through the dataset for a specified number of epochs, performing forward and backward passes to optimize the model.

o	Validation Loop: Evaluates the model on the validation dataset to monitor performance and prevent overfitting.

**4.	Evaluation:**
o	Loss Tracking: Both training and validation losses are tracked and plotted to visualize the model's performance over epochs.

