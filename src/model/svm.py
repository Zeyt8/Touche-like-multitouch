import numpy as np
from scipy.signal import savgol_filter
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations, chain

def powerset(iterable):
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

test_cases = list(powerset(range(1, 5)))
X_train_clean = np.loadtxt('data_train.txt', delimiter=' ')
X_test = np.loadtxt('data_test.txt', delimiter=' ')

def smooth(profile, window=9, polyorder=2):
    window = window if window % 2 == 1 else window + 1
    return savgol_filter(profile, window, polyorder)

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

def augment_with_synthetic_noise(X_clean, noise_std, n_repeats=15, seed=0):
    rng = np.random.default_rng(seed)
    X_aug = []
    label_idx = []
    for i, row in enumerate(X_clean):
        for _ in range(n_repeats):
            noisy = row + rng.normal(0, noise_std, size=row.shape)
            X_aug.append(noisy)
            label_idx.append(i)
    return np.array(X_aug), np.array(label_idx)

X_train_aug, train_label_idx = augment_with_synthetic_noise(
    X_train_clean, noise_std=0.02, n_repeats=15
)

X_feat = np.array([extract_features(row) for row in X_train_aug])

y = np.zeros((len(X_train_aug), 4))
for i, idx in enumerate(train_label_idx):
    for node in test_cases[idx]:
        y[i, node - 1] = 1

node_classifiers = {}
for node in range(1, 5):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        #('pca', PCA(n_components=0.9)),
        ('svm', SVC(kernel='rbf', C=2, gamma="scale")),
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

