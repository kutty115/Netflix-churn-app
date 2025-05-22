Netflix Churn Prediction App* is a machine learning web application built with *Flask* that predicts whether a user is likely to churn (unsubscribe) based on key usage metrics such as:  
* Monthly hours watched
* Data usage (in GB)
* Subscription plan price

The model is trained using a neural network in *TensorFlow/Keras*, with early stopping based on validation accuracy to avoid overfitting. Input features are scaled using StandardScaler, and the final model is saved for real-time predictions via a web interface.

This project demonstrates:

* Data preprocessing with pandas and scikit-learn
* Binary classification with deep learning
* Model training with early stopping
* Web deployment using Flask
* Saving/loading models and scalers for production use

