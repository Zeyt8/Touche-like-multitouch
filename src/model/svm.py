import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from itertools import combinations, chain

def powerset(iterable):
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

test_cases = list(powerset(range(1, 5)))
X_train = np.loadtxt('data_train.txt', delimiter=' ')
X_test = np.loadtxt('data_test.txt', delimiter=' ')

def smooth(profile, window=5):
    kernel = np.ones(window) / window
    return np.convolve(profile, kernel, mode='same')

def extract_features(profile):
    profile = smooth(np.array(profile))
    derivative_kernel = np.array([-1, 1])
    derivative_features = []

    for n_points in [10, 20, 40]:
        downsampled = np.interp(
            np.linspace(0, len(profile) - 1, n_points),
            np.arange(len(profile)),
            profile
        )
        derivative = np.convolve(downsampled, derivative_kernel, mode='same')
        derivative_features.append(derivative)

    minima = np.array([np.min(profile)])

    return np.concatenate([profile] + derivative_features + [minima])

X_feat = np.array([extract_features(row) for row in X_train])

y = np.zeros((len(test_cases), 4))
for i, combo in enumerate(test_cases):
    for node in combo:
        y[i, node - 1] = 1

node_classifiers = {}
for node in range(1, 5):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=3, gamma=0.01)),
    ])
    pipe.fit(X_feat, y[:, node - 1])
    node_classifiers[node] = pipe

def detect_touch(profile, max_nodes=4):
    features = extract_features(profile).reshape(1, -1)

    node_scores = {}
    for node in range(1, max_nodes + 1):
        score = node_classifiers[node].decision_function(features)[0]
        node_scores[node] = score
 
    print(f"  Raw SVM scores: { {f'n_{k}': str(round(v, 3)) for k, v in node_scores.items()} }")
 
    candidates = [node for node, score in node_scores.items() if score > 0]
    return candidates

print("=== Verification across all combinations ===")
for i, combo in enumerate(test_cases):
    label = combo if combo else '(-)'
    print(f"\nSample {label}:")
    result = detect_touch(X_test[i])
    match = "OK" if set(result) == set(combo) else "FAILED"
    print(f"  Detected nodes: {result}  [{match}]")

