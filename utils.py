import numpy as np


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def batch_generator(X, y, batch_size):
    # why use .copy() ?
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    # or:
    #   from sklearn.utils import shuffle
    #   X, y = shuffle(X, y, random_state=0)

    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i: i + batch_size], y_copy[i: i + batch_size]
            i += batch_size
        else:
            i = 0
            # just set i = 0, it still works,
            # so why do shuffle again?
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


if __name__ == "__main__":

    X = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])

    gen = batch_generator(
        X,
        y,
        2
    )

    for _ in range(8):
        x, y = next(gen)
        print(x, y)


