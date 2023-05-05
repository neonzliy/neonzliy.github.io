---
layout: post
title: Some Toy Algorithms - Sentiment Classification
subtitle: An attempt to implement common used models from scratch 
---

This is an attempt to do a sentiment classification for the movies. We will build a binary linear
classifier that reads movie reviews, as you would see from the sites like Rotten Tomatoes, and predicts whether the
result is positive or negative

The attempt is to classified given movie reviews as ”positive” and ”negative”, Here, we will create a simple text classification
system that can perform this task automatically.

Here's some preperation:

```
import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]
```

Binary classification

```
def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass

    import string

    # Convert the review to lowercase
    x = x.lower()

    # Remove punctuation from the text
    x = x.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the text (split into words)
    words = x.split()

    # Create an empty dictionary to store word features
    word_features = {}

    # Iterate through the words and add them to the dictionary with their counts
    for word in words:
        if word in word_features:
            word_features[word] += 1
        else:
            word_features[word] = 1

    return word_features
```

Stochastic Gradient Descent:

```
T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    
    weights = {}  # feature => weight

    def dotProduct(d1, d2):
        """
        @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
        @param dict d2: same as d1
        @return float: the dot product between d1 and d2
        """
        if len(d1) < len(d2):
            return dotProduct(d2, d1)
        else:
            return sum(d1.get(f, 0) * v for f, v in d2.items())

    weights = {}  # feature => weight

    for epoch in range(numEpochs):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            dot_prod = dotProduct(weights, phi)

            if y * dot_prod <= 1:
                for feature in phi:
                    if feature not in weights:
                        weights[feature] = 0
                    weights[feature] += eta * y * phi[feature]

        train_error = evaluatePredictor(
            trainExamples,
            lambda x: 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1,
        )
        validation_error = evaluatePredictor(
            validationExamples,
            lambda x: 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1,
        )

        print(
            f"Epoch {epoch + 1}: Training error {train_error}, Validation error {validation_error}"
        )

    return weights
```

Create an toy dataset using generateExample function and use as a test case for learnPredictor.

```
def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        while True:
            phi = {}
            for key in weights.keys():
                if (
                    random.random() < 0.5
                ):  # Randomly select a subset of keys from weights
                    phi[key] = random.randint(
                        -10, 10
                    )  # Assign random integer values to the selected keys

            score = dotProduct(phi, weights)
            if score != 0:
                break

        y = 1 if score >= 0 else -1

        # Test that the randomly generated example coincides with the given weights
        predicted_y = 1 if dotProduct(weights, phi) >= 0 else -1
        assert (
            predicted_y == y
        ), f"Predicted label {predicted_y} does not match the generated label {y}"

        return (phi, y)

    return [generateExample() for _ in range(numExamples)]
```
Some more character extraction to handel edge cases

```
def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:

    def extract(x):
        pass

        x = x.replace(" ", "")  # Remove spaces from the input string
        features = {}

        # Iterate through the string and count the occurrences of each n-gram
        for i in range(len(x) - n + 1):
            ngram = x[i : i + n]
            if ngram in features:
                features[ngram] += 1
            else:
                features[ngram] = 1

        return features

    return extract
```
Finally, a test function to test different values of n for extractCharacterFeatures

```
def testValuesOfN(n: int):
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )
```
