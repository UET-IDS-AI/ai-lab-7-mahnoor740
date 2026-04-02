import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

# Q1 Naive Bayes

def naive_bayes_mle_spam():

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # 1. Tokenization
    tokenized_texts = [text.split() for text in texts]

    # 2. Vocabulary
    vocab = set()
    for tokens in tokenized_texts:
        vocab.update(tokens)

    # 3. Class Priors
    priors = {
        1: np.sum(labels == 1) / len(labels),
        0: np.sum(labels == 0) / len(labels)
    }

    # 4. Word counts per class
    word_counts = {
        1: {word: 0 for word in vocab},
        0: {word: 0 for word in vocab}
    }

    total_words = {1: 0, 0: 0}

    for tokens, label in zip(tokenized_texts, labels):
        for word in tokens:
            word_counts[label][word] += 1
            total_words[label] += 1

    # 5. Word probabilities (MLE, no smoothing)
    word_probs = {
        1: {},
        0: {}
    }

    for c in [0, 1]:
        for word in vocab:
            if total_words[c] > 0:
                word_probs[c][word] = word_counts[c][word] / total_words[c]
            else:
                word_probs[c][word] = 0

    # 6. Prediction
    test_tokens = test_email.split()

    scores = {}
    for c in [0, 1]:
        score = np.log(priors[c])
        for word in test_tokens:
            prob = word_probs[c].get(word, 0)
            if prob == 0:
                score = -np.inf
                break
            score += np.log(prob)
        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction

# Q2 KNN

def knn_iris(k=3, test_size=0.2, seed=0):

    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 3. Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # 4. Prediction function
    def predict(X1, X2, y2):
        predictions = []
        for x in X1:
            distances = [euclidean_distance(x, x_train) for x_train in X2]
            k_indices = np.argsort(distances)[:k]
            k_labels = y2[k_indices]

            # Majority vote
            values, counts = np.unique(k_labels, return_counts=True)
            pred = values[np.argmax(counts)]
            predictions.append(pred)

        return np.array(predictions)

    # 5. Predictions
    train_preds = predict(X_train, X_train, y_train)
    test_preds = predict(X_test, X_train, y_train)

    # 6. Accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy, test_preds
