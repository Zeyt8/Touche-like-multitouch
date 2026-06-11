import numpy as np
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

X_raw = np.loadtxt('data.txt', delimiter=' ')[:5]
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

clf = Pipeline([
    ("classifier", SVC(
        kernel='poly',
        C=2.0,
        coef0=1.0,
        gamma='scale',
        decision_function_shape='ovr',  # one score per class
    ))
])

clf.fit(X, y)

svm = clf.named_steps["classifier"]
print("Classes:", svm.classes_)
print("Support vectors per class:", svm.n_support_)

for i in range(len(X)):
    sample = X[i].reshape(1, -1)
    scores = clf.decision_function(sample)[0]   # 5 scores, one per class
    pred = clf.predict(sample)[0]
    print(f"\nSample {i} (true label: {y[i]})")
    print(f"  Scores: { {f'n{j}':str(round(s, 3)) for j, s in enumerate(scores)} }")
    print(f"  Predicted: node {pred}")

def detect_touch(profile):
    features = extract_features(profile).reshape(1, -1)
    scores = clf.decision_function(features)[0]   # raw SVM margins per class
    predicted_class = clf.predict(features)[0]
    return {
        "predicted_node": int(predicted_class),
        "scores": {f"node_{j}": round(float(s), 4) for j, s in enumerate(scores)}
    }

idx = np.random.choice(len(X_raw))
print(f"\nTesting sample index: {idx} (true label: {y[idx]})")
result = detect_touch(X_raw[idx])
print(f"Predicted node: {result['predicted_node']}")
print(f"Scores: {result['scores']}")
