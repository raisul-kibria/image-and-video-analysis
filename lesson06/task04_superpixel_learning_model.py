import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_and_test_model():
    """ Trains and tests a model to classify superpixels. """
    # Read sample image
    img_bgr = cv2.imread('../samples/Ki67.jpg', cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25)
    # Infer (automatically) which pixels will be set as positives.
    ground_truth = cv2.dilate((cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) <= 128).astype('uint8') * 255, np.ones((3, 3)))

    # Superpixel segmentation
    sh = img_bgr.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(image_width=sh[1], image_height=sh[0], image_channels=sh[2], num_superpixels=150, num_levels=4)
    seeds.iterate(img=img_bgr, num_iterations=5)
    region_labels = seeds.getLabels()

    # Get features
    # X = get_geometric_features(region_labels)
    # X = get_photometric_features(img_bgr, region_labels)
    X = np.concatenate([get_geometric_features(region_labels), get_photometric_features(img_bgr, region_labels)], axis=1)

    # Get labels
    positives_per_region = []
    for idx_l in range(np.max(region_labels) + 1):
        region_mask = (region_labels == idx_l)
        positives_per_region.append(
            np.mean(ground_truth[region_mask])
        )
    y = np.array(positives_per_region) > 0.5

    # Create model
    # model = LogisticRegression()
    model = RandomForestClassifier()

    # Train model
    # Your code here: use `model.fit(...)` to train the model.
    # ...
    model.fit(X, y)
    # Make predictions
    # Your code here: use `model.predict(X)` to train the model.
    # ...
    predictions = model.predict(X) # np.zeros(img_bgr.shape[0:2])
    pred_mask = np.zeros(img_bgr.shape[0:2])
    for label_idx in range(np.max(region_labels) + 1):
        region_mask = (region_labels == label_idx)
        pred_mask[region_mask] = int(predictions[label_idx])

    predictions = pred_mask
    # Better visualizations
    border_mask = region_labels != cv2.erode(region_labels.astype('uint8'), np.ones((3, 3)))
    img_with_borders = np.copy(img_bgr)
    img_with_borders[border_mask, ...] = (255, 0, 0)
    img_with_gt = np.copy(img_bgr)
    img_with_gt[ground_truth == 255] = (0, 0, 255)
    img_with_gt[border_mask, ...] = (255, 0, 0)
    img_with_pred = np.copy(img_bgr)
    img_with_pred[predictions == 1] = (0, 255, 0)
    img_with_pred[border_mask, ...] = (255, 0, 0)

    _, axs = plt.subplots(2, 2)
    [ax.axis('off') for ax in axs.flatten()]
    axs[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 1].imshow(cv2.cvtColor(img_with_borders, cv2.COLOR_BGR2RGB))
    axs[1, 0].imshow(cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Ground truth')
    axs[1, 1].imshow(cv2.cvtColor(img_with_pred, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Predictions')
    plt.show()


def get_geometric_features(region_labels:np.ndarray) -> np.ndarray:
    """ Computes geometric features for each region in the image. """
    features = []
    for label_idx in range(np.max(region_labels) + 1):
        region_mask = (region_labels == label_idx)

        # Compute geometric features for this region.
        area = np.sum(region_mask)
        perimeter = np.sum(region_mask != cv2.erode(region_mask.astype('uint8'), np.ones((3, 3))))
        pixel_locations = np.where(region_mask)
        centroid_x = np.mean(pixel_locations[0])
        centroid_y = np.mean(pixel_locations[1])
        std_x = np.std(pixel_locations[0])
        std_y = np.std(pixel_locations[1])
        roundness = 4 * np.pi * area / (perimeter**2)

        # Store them
        features.append([
            area,
            perimeter,
            centroid_x,
            centroid_y,
            std_x,
            std_y,
            roundness,
        ])

    # Output as a numpy array indexed as [region_idx, feature_idx].
    X = np.array(features)
    return X


def get_photometric_features(img_bgr: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    """ Computes photometric features for each region in the image. """
    features = []
    for label_idx in range(np.max(region_labels) + 1):
        region_mask = (region_labels == label_idx)

        # Compute photometric features for this region.
        max_red_value = img_bgr[region_mask, 2].max()
        # Your code here: Add others (e.g. other channels, mean values, std of values, ...)
        max_green_value = img_bgr[region_mask, 1].max()
        max_blue_value = img_bgr[region_mask, 0].max()
        
        mean_r = np.mean(img_bgr[region_mask,2])
        std_val_r = np.std(img_bgr[region_mask,2])

        mean_g = np.mean(img_bgr[region_mask,1])
        std_val_g = np.std(img_bgr[region_mask,1])

        mean_b = np.mean(img_bgr[region_mask,0])
        std_val_b = np.std(img_bgr[region_mask,0])

        mean = np.mean(img_bgr[region_mask,:])
        std_val = np.std(img_bgr[region_mask,:])

        # Store them
        features.append([
            max_red_value,
            max_green_value,
            max_blue_value,
            mean_r,
            std_val_r,
            mean_g,
            std_val_g,
            mean_b,
            std_val_b,
            mean,
            std_val
            #
            #  ...
        ])

    # Output as a numpy array indexed as [region_idx, feature_idx].
    X = np.array(features)
    return X


if __name__ == '__main__':
    train_and_test_model()
