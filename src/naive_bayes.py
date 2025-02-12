import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1] 
        self.conditional_probabilities = self.estimate_conditional_probabilities(features,labels,delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # Count number of samples for each output class and divide by total of samples
        class_priors: Dict[int, torch.Tensor] = {}

        for label in labels:
            label = int(label)
            if label in class_priors:
                class_priors[label] += 1
            else:
                class_priors[label] = 1
        
        for key in class_priors:
            class_priors[key] = torch.tensor(class_priors[key]/len(labels), dtype=torch.float32)

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # Estimate conditional probabilities for the words in features and apply smoothing
        class_word_counts: Dict[int, torch.Tensor] = {}
        total_word_counts: Dict[int, torch.Tensor] = {}

        for index, label in enumerate(labels):
            label = int(label)
            if label not in total_word_counts:
                total_word_counts[label] = features[index].sum()
            else:
                total_word_counts[label] += features[index].sum()
        
        for index, label in enumerate(labels):
            label = int(label)
            if label not in class_word_counts:
                class_word_counts[label] = (features[index] + delta) / (total_word_counts[label] + delta * self.vocab_size)
            else:
                class_word_counts[label] += features[index] / (total_word_counts[label] + delta * self.vocab_size)
        
        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # Calculate posterior based on priors and conditional probabilities of the words
        log_posteriors = {}
        for label, prior in self.class_priors.items():
            conditional_prob = self.conditional_probabilities[label]
            log_posteriors[label] = torch.log(prior) + torch.sum(feature * torch.log(conditional_prob))

        return torch.tensor([log_posteriors[label] for label in sorted(log_posteriors.keys())], dtype=torch.float32)

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # Calculate log posteriors and obtain the class of maximum likelihood 
        probs = self.predict_proba(feature)
        pred: int = int(torch.argmax(probs))
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # Calculate log posteriors and transform them to probabilities (softmax)
        posteriors = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.softmax(posteriors, dim=0)
        return probs
