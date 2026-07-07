import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from itertools import combinations, chain

X_raw = np.loadtxt('data.txt', delimiter=' ')[:5]
X_multitouch = np.loadtxt('data.txt', delimiter=' ')[5:]
y = np.array([0, 1, 2, 3, 4])

def extract_features(profile):
    profile = np.array(profile)
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

X = np.array([extract_features(row) for row in X_raw])

clf = SVC(
        kernel='poly',
        C=2.0,
        coef0=1.0,
        gamma='scale',
        decision_function_shape='ovr',
    )

clf.fit(X, y)

def detect_touch(profile, max_nodes=4):
    features = extract_features(profile)

    scores = clf.decision_function(features.reshape(1, -1))[0]
    print(f"  Background score: {round(scores[0], 3)}")
    node_scores = {node: scores[node] for node in range(1, max_nodes + 1)}
    print(f"  Raw SVM scores: { {f'n_{k}': str(round(v, 3)) for k, v in node_scores.items()} }")

    candidates = [node for node, score in node_scores.items() if score > 0]

    return candidates

print("=== Single-touch verification ===")
for i in range(5):
    print(f"\nSample {i}:")
    result = detect_touch(X_raw[i])
    print(f"  Detected nodes: {result}")

def powerset(iterable):
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

test_cases = list(powerset(range(1, 5)))[5:]

print("=== Multi-touch verification ===")
for i in range(len(X_multitouch)):
    print(f"\nSample {[x for x in test_cases[i]]}:")
    result = detect_touch(X_multitouch[i])
    print(f"  Detected nodes: {result}")
