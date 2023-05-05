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

~~~
import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]
~~~

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

{% highlight python linenos %}
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
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
