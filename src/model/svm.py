import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

X_raw = np.loadtxt('data.txt', delimiter=' ')
y = np.arange(len(X_raw))

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
)

clf.fit(X, y)

def detect_multitouch(profile, threshold=0.8):
    features = extract_features(profile).reshape(1, -1)
    similarities = cosine_similarity(features, X)[0]
    touched = np.where(similarities >= threshold)[0]
    return touched.tolist()

idx = np.random.choice(len(X_raw), size=2, replace=False)
print(f"Testing with touch points: {idx}")
incoming_vector = X_raw[idx[0]] + X_raw[idx[1]]
result = detect_multitouch(incoming_vector, threshold=0.5)
print(f"Detected touch points: {result}")
