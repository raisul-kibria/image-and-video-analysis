import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_and_test_model():
    img = cv2.imread('../samples/Retinal_DRIVE21_original.tif', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('../samples/Retinal_DRIVE21_gt.tif', cv2.IMREAD_GRAYSCALE)

    # Get features
    # X = features_gabor_filter_bank(img)
    # X = features_eigenvalues_hessian(img)
    X = np.concatenate([features_gabor_filter_bank(img), features_eigenvalues_hessian(img)], axis=1)

    # Get labels
    num_pixels = img.shape[0] * img.shape[1]
    y = mask.flatten()/255

    # Create model
    # model = LogisticRegression()
    model = RandomForestClassifier()

    # Train model
    random_selection_positives = np.random.choice(np.where(y == 1)[0], 5000, replace=False)
    random_selection_negatives = np.random.choice(np.where(y == 0)[0], 5000, replace=False)
    random_selection = np.concatenate([random_selection_positives, random_selection_negatives])
    X_train = X[random_selection, :]
    y_train = y[random_selection]
    # Your code here: use `model.fit(X, y)` to train the model.
    # ...
    model.fit(X_train, y_train)

    # Make predictions
    # Your code here: use `model.predict(X)` to train the model.
    # ...

    predictions = model.predict(X)  # You may delete this line.
    print(predictions.shape)
    predictions = predictions.reshape(img.shape) > 0.5
    # Show results
    img_with_gt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_with_gt[mask == 255] = [0, 0, 255]
    img_with_pred = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_with_pred[predictions == 1] = [0, 255, 0]

    _, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 1].set_axis_off()
    axs[1, 0].imshow(img_with_gt)
    axs[1, 1].imshow(img_with_pred)
    plt.show()


def features_gabor_filter_bank(img):
    """Computes features based on Gabor filters."""
    # (This function is already provided completely.)
    kernels = [
        cv2.getGaborKernel(ksize=(15, 15), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=0)
        for sigma in [3, 5, 7]
        for theta in [np.pi, np.pi / 2, 0]
        for lambd in [1.5, 2]
        for gamma in [1, 1.5]
    ]
    filtered_images = [cv2.filter2D(img, cv2.CV_64F, kernel) for kernel in kernels]

    # Create features
    X = np.stack([f.flatten() for f in filtered_images], axis=-1)
    print("gabor", X.shape)
    return X


def features_eigenvalues_hessian(img):
    """Computes features based on the eigenvalues of the Hessian matrix."""
    # Your code here
    H0 = cv2.Sobel(img, cv2.CV_32F, dx = 2, dy = 0)
    H1 = cv2.Sobel(img, cv2.CV_32F, dx = 1, dy = 1)
    H2 = cv2.Sobel(img, cv2.CV_32F, dx = 1, dy = 1)
    H3 = cv2.Sobel(img, cv2.CV_32F, dx = 0, dy = 2)
    trace = H0 + H3
    det = H0 * H3 - H1 * H2
    x1 = (trace + np.sqrt(np.power(trace, 2) - 4 * det)) / 2
    x2 = (trace - np.sqrt(np.power(trace, 2) - 4 * det)) / 2
    
    filtered_images = [H0, H1, H2, H3, det, trace, x1, x2]
    features = np.stack([f.flatten() for f in filtered_images], axis=-1)
    print("eigen", features.shape)
    return features

if __name__ == '__main__':
    train_and_test_model()
