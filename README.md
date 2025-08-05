# Speech-Recognition-with-Hidden-Markov-Models
Speech Recognition with Hidden Markov Models is a machine learning project focused on building a voice-based classifier that identifies spoken instances of daily food-related words such as “Rice”, “Pizza”, “Burger”, “Sandwich”, “Sausage”, “Meatball”, and “Spaghetti”.

The recognizer uses Mel-Frequency Cepstral Coefficients (MFCC) to extract features from speech audio and then applies Hidden Markov Models (HMMs)—trained via the Baum-Welch algorithm—to classify input speech signals.

Key Project Steps:
 - Data Preparation: Curated and recorded speech samples for seven distinct food-related words. Divided data into training and testing sets.
 - Feature Extraction: Implemented MFCC-based feature extraction to transform raw audio into meaningful signal representations.
 - Model Training: Trained individual HMMs for each word using the Baum-Welch algorithm (an Expectation-Maximization approach).
 - Model Evaluation: Classified test audio by selecting the HMM that best fits the input sequence.
Advanced Exploration:
 - Investigated the impact of hyperparameters (number of MFCCs, hidden states, and observation symbols) on model performance.
 - Bonus Analysis: Compared HMM performance with alternate classifiers like SVM, Naive Bayes, and Neural Networks, discussing the pros and cons of EM-based HMM training versus discriminative approaches.

This project offered a comprehensive, end-to-end introduction to speech signal processing, HMM-based modeling, and practical challenges in acoustic classification.
