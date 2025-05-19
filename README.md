# Fashion MNIST Classification Using CNN

## Project Overview  
This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. Fashion MNIST contains 70,000 grayscale images of 10 different types of clothing items, each image sized 28x28 pixels.

The model learns to distinguish categories such as T-shirts, trousers, dresses, sneakers, bags, and more by extracting visual features using multiple convolutional layers and classifying them into one of the 10 classes.

---

## Dataset Description  
- **Dataset:** Fashion MNIST  
- **Number of training images:** 60,000  
- **Number of test images:** 10,000  
- **Image size:** 28x28 pixels, grayscale  
- **Classes:** 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

---

## Data Preprocessing  
- Images are reshaped to include a channel dimension (28x28x1) to match CNN input requirements.  
- Pixel values are normalized from the range [0, 255] to [0, 1] to improve model training stability.  
- Sample images with their class labels are visualized to understand the dataset.

---

## CNN Model Architecture  
The CNN model consists of:  
- **Conv2D layers:** Extract spatial features with increasing depth (64, 128, 256 filters).  
- **Batch Normalization:** Helps stabilize and speed up training.  
- **MaxPooling2D:** Reduces spatial dimensions, lowering computation and capturing dominant features.  
- **GlobalAveragePooling2D:** Averages each feature map, reducing parameters before dense layers.  
- **Dense layers:** Fully connected layers for classification.  
- **Dropout:** Regularizes the model to prevent overfitting.  
- **Softmax activation:** Outputs class probabilities.

---

## Training  
- Optimizer: Adam  
- Loss function: Sparse categorical cross-entropy  
- Epochs: 10  
- Validation performed on test set during training to monitor performance.

---

## Evaluation and Results  
- The model is evaluated on the test dataset, achieving high classification accuracy.  
- Confusion matrix is plotted to visualize class-wise performance.  
- A detailed classification report with precision, recall, and F1-score for each class is generated.  

---

## Model Visualization  
- The model summary provides an overview of each layer and the number of parameters.  
- The architecture is saved as an image (`model_architecture.png`) for easy reference.

---

## Additional Notes  
- Variable naming: Avoid using Python built-in names such as `type` as variable names to prevent errors.  
- Example code snippets demonstrate safe handling of data types and input shapes.  
- The project uses standard libraries such as TensorFlow, NumPy, Matplotlib, Seaborn, and scikit-learn.

---

## Applications  
- Automated clothing item recognition for retail or e-commerce.  
- Fashion recommendation systems.  
- Educational purposes to understand CNNs and image classification.

---

## How to Run  
1. Install required packages (`tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).  
2. Load the Fashion MNIST dataset using TensorFlow.  
3. Preprocess images by reshaping and normalization.  
4. Build and compile the CNN model as described.  
5. Train the model on the training data and validate using test data.  
6. Evaluate and analyze performance using confusion matrix and classification report.

---

## License  
This project is open-source and free to use for educational and research purposes.

---


