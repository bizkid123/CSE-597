import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error


def train_test_split(
    base_set,
    adversarial_set,
    percent_train=0.8,
    auto_balance=True,
    classification=True,
    column="logit_diffs",
    seed=0,
):
    # Training set is first 80% of base examples and first 80% of adversarial examples
    # Test set is last 20% of base examples and last 20% of adversarial examples
    # Randomly shuffle train and test sets using seed
    np.random.seed(seed)

    base_set = base_set.sample(frac=1).reset_index(drop=True)
    adversarial_set = adversarial_set.sample(frac=1).reset_index(drop=True)

    base_train_size = int(len(base_set) * percent_train)
    adversarial_train_size = int(len(adversarial_set) * percent_train)

    base_test_size = len(base_set) - base_train_size
    adversarial_test_size = len(adversarial_set) - adversarial_train_size

    if auto_balance:
        base_train_size = min(base_train_size, adversarial_train_size)
        base_test_size = min(base_test_size, adversarial_test_size)

        adversarial_train_size = base_train_size
        adversarial_test_size = base_test_size

    # Get training set
    base_train = base_set[column][:base_train_size]
    adversarial_train = adversarial_set[column][:adversarial_train_size]

    # Select first half of base train and second half of adversarial train
    base_train = base_train[: int(len(base_train) / 2)]
    adversarial_train = adversarial_train[int(len(adversarial_train) / 2) :]

    # concatenate and process
    train_x = pd.concat([base_train, adversarial_train])
    try:
        train_x = [list(x) for x in train_x]
    except TypeError:
        pass
    if classification:
        train_y = [0] * len(base_train) + [1] * len(adversarial_train)
    else:
        train_y = (
            base_set["confidence"][: len(base_train)].tolist()
            + adversarial_set["confidence"][: len(adversarial_train)].tolist()
        )

    # Get test set
    base_test = base_set[column][len(base_train) :]
    adversarial_test = adversarial_set[column][len(adversarial_train) :]

    if auto_balance:
        test_length = min(base_test_size, adversarial_test_size)
        base_test = base_test[:test_length]
        adversarial_test = adversarial_test[:test_length]

    print(
        "Number of base train examples: {}, number of adversarial train examples: {}".format(
            len(base_train), len(adversarial_train)
        )
    )
    print(
        "Number of base test examples: {}, number of test adversarial examples: {}".format(
            len(base_test), len(adversarial_test)
        )
    )

    # concatenate and process
    test_x = pd.concat([base_test, adversarial_test])
    try:
        test_x = [list(x) for x in test_x]
    except TypeError:
        pass
    # test_x = [list(x) for x in test_x]
    if classification:
        test_y = [0] * len(base_test) + [1] * len(adversarial_test)
    else:
        test_y = (
            base_set["confidence"][len(base_train) :].tolist()
            + adversarial_set["confidence"][len(adversarial_train) :].tolist()
        )

    return train_x, train_y, test_x, test_y


def train_defense_classifier(train_x, train_y, classification=True):
    if classification:
        # Train svm
        clf = svm.SVC()
    else:
        # Train svm
        clf = svm.SVR()

    clf.fit(train_x, train_y)

    return clf


def test_model(clf, test_x, test_y, classification=True, verbose=True):
    # Test svm
    predictions = clf.predict(test_x)

    # convert to numpy array
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    base_x = test_x[test_y == 0]
    base_y = test_y[test_y == 0]

    adversarial_x = test_x[test_y == 1]
    adversarial_y = test_y[test_y == 1]

    base_predictions = predictions[test_y == 0]
    adversarial_predictions = predictions[test_y == 1]

    overall_accuracy = sum(predictions == test_y) / len(test_y)
    if verbose:
        if classification:
            # Get accuracy
            try:
                base_accuracy = sum(base_predictions == base_y) / len(base_y)
                print("Base accuracy: {}".format(base_accuracy))
            except ZeroDivisionError:
                pass

            try:
                adversarial_accuracy = sum(
                    adversarial_predictions == adversarial_y
                ) / len(adversarial_y)

                print("Adversarial accuracy: {}".format(adversarial_accuracy))
            except ZeroDivisionError:
                pass
            print(
                "Overall accuracy: {}".format(sum(predictions == test_y) / len(test_y))
            )
        else:
            # Difference between predictions and actual values
            # Mean squared error
            base_mse = mean_squared_error(base_predictions, base_y)
            print("Base MSE: {}".format(base_mse))

            adversarial_mse = mean_squared_error(adversarial_predictions, adversarial_y)
            print("Adversarial MSE: {}".format(adversarial_mse))

            print("Overall MSE: {}".format(mean_squared_error(predictions, test_y)))

    return predictions, overall_accuracy


# Baseline using only first logit difference
def threshold_baseline(
    test_x, test_y, threshold, index=0, exclude_zeros=True, verbose=False
):
    correct_base = 0
    correct_adversarial = 0
    total_base = 0
    total_adversarial = 0
    for i in range(len(test_x)):
        if exclude_zeros and test_x[i][index] == 0:
            continue
        if test_x[i][index] <= threshold:
            if test_y[i] == 1:
                correct_adversarial += 1
                total_adversarial += 1
            else:
                total_base += 1
        else:
            if test_y[i] == 0:
                correct_base += 1
                total_base += 1
            else:
                total_adversarial += 1

    if verbose:
        if total_base == 0:
            print("No base examples")
        else:
            print("Base accuracy: {}".format(correct_base / total_base))

        if total_adversarial == 0:
            print("No adversarial examples")
        else:
            print(
                "Adversarial accuracy: {}".format(
                    correct_adversarial / total_adversarial
                )
            )
        print(
            "correct_base: {}, correct_adversarial: {}, total_base: {}, total_adversarial: {}".format(
                correct_base, correct_adversarial, total_base, total_adversarial
            )
        )

    return (correct_base + correct_adversarial) / (total_base + total_adversarial)


def get_best_threshold(train_x, train_y, index=0, abs_precision=0.0001):
    # Ternary search for best threshold
    low = min([x[index] for x in train_x])
    high = max([x[index] for x in train_x])
    while abs(high - low) > abs_precision:
        # print(low, high)
        low_third = low + (high - low) / 3
        high_third = high - (high - low) / 3

        if threshold_baseline(train_x, train_y, low_third, index) < threshold_baseline(
            train_x, train_y, high_third, index
        ):
            low = low_third
        else:
            high = high_third

    best_threshold = (low + high) / 2

    return best_threshold


def confidence_threshold(test_x, test_y, threshold):
    correct_base = 0
    correct_adversarial = 0
    total_base = 0
    total_adversarial = 0
    for i in range(len(test_x)):
        # print(test_x[i], type(test_x[i]))
        if test_x.iloc[i] <= threshold:
            if test_y[i] == 1:
                correct_adversarial += 1
                total_adversarial += 1
            else:
                total_base += 1
        else:
            if test_y[i] == 0:
                correct_base += 1
                total_base += 1
            else:
                total_adversarial += 1

    return (correct_base + correct_adversarial) / (total_base + total_adversarial)


def get_best_confidence_threshold(train_x, train_y, abs_precision=0.0001):
    # Ternary search for best threshold
    low = min(train_x)
    high = max(train_x)
    print(f"low: {low}, high: {high}")
    while abs(high - low) > abs_precision:
        # print(low, high)
        low_third = low + (high - low) / 3
        high_third = high - (high - low) / 3

        if confidence_threshold(train_x, train_y, low_third) < confidence_threshold(
            train_x, train_y, high_third
        ):
            low = low_third
        else:
            high = high_third

    best_threshold = (low + high) / 2

    return best_threshold
