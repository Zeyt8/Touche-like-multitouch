import numpy as np
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

X_raw = np.loadtxt('data.txt', delimiter=' ')
y = np.array([
    [0,0,0,0],  # no touch
    [1,0,0,0],  # node 0
    [0,1,0,0],  # node 1
    [0,0,1,0],  # node 2
    [0,0,0,1],  # node 3
])

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

base_svm = SVC(
    kernel='poly',
    C=2.0,
    coef0=1.0,
    gamma='scale',
)

clf = Pipeline([
    ("classifier", MultiOutputClassifier(base_svm))
])

clf.fit(X, y)

for i, estimator in enumerate(clf.named_steps["classifier"].estimators_):
    print(f"\nNODE {i}")
    print("classes:", estimator.classes_)
    print("support vectors:", estimator.support_vectors_.shape)
    print("dual coef:", estimator.dual_coef_)
    print("intercept:", estimator.intercept_)

for i in range(len(X)):
    sample = X[i].reshape(1, -1)

    print(f"\nSample {i}")

    for j, estimator in enumerate(clf.named_steps["classifier"].estimators_):
        score = estimator.decision_function(sample)[0]
        pred = estimator.predict(sample)[0]

        print(f"  Node {j}: score={score:.4f}, pred={pred}")

def detect_multitouch(profile, threshold=0.8):
    features = extract_features(profile).reshape(1, -1)
    probabilities = np.array([
        estimator.decision_function(features)[0]
        for estimator in clf.named_steps["classifier"].estimators_
    ])
    return probabilities

idx = np.random.choice(len(X_raw), size=1, replace=False)
print(f"Testing with touch points: {idx}")
incoming_vector = X_raw[idx[0]]
result = detect_multitouch(incoming_vector, threshold=0.1)
print(f"Detected touch points: {result}")
