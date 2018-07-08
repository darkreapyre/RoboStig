# Data Preprocessing
One of the most important steps in any Machine Learning Pipeline is the *Data Preprocessing* step. This is where you ensure that you have the right data, that is correctly formatted and usable for training. The typical tasks that can take place during this step are:
- Removing or inferring missing data.
- Extracting the needed features from the data and removing unnecessary data..
- Inferring or creating more data.

In this next section, you will be implementing some of these tasks.

## Feature Extraction

## Image Transformations
Along with extracting the necessary features from the data, a similar process should also be applied to the images.

__Exercise:__ For your first exercise, you will need to implement the following image transformation functions:
1. Crop images for to only focus on the Region of Interest (ROI).
    <div class="alert alert-danger">
        <strong>Tip: </strong>The input images from the simulator are in RGB format and represented as a `numpy`.array` of format (Height, Width, Channel. Extract just the parts of the height array and width array for the region of interest.
    </div>
2. Resize images to $66 \times 200 \times 3$.
    <div class="alert alert-danger">
        <strong>Tip: </strong>The `cv2.resize()` function may be helpful.
    </div>
3. Convert to YUV Channels.
    <div class="alert alert-danger">
        <strong>Tip: </strong>The `cv2.cvtColor()` function may be helpful.
    </div>

<details><summary><b>Solution (Click to Expand)</b></summary><p>

```
# Image Transformation: Crop
def crop(image):
    """
    Crops the image by emoving the sky at the top and the car front at the bottom.

    Arguments:
    image -- numpy.array representing an RGB image of format (Height, Width, Channel)
   
    Returns:
    Cropped image.
    """
    return image[60:-25, :, :]
```
</p>
</details>


---

# Re-code for Augmentation Pipeline

>__Note to self:__ It may be a good idea to hard-code changing the images to "channel first" in the hopes of speeding up MXNet. Should this be implemented, then the image shape must be adjusted before the `return` statement in the `transform()` function as follows:


## Origional Function Caller:

```
X_train, y_train = aug_pipeline('data', X_train, y_train, len(X_train), True)
X_valid, y_valid = aug_pipeline('data', X_valid, y_valid, len(X_valid), False)
```

## Original Function:

```
def aug_pipeline(data_dir, image_paths, steering_angles, batch_size, is_training):
    # Generate training image give image paths and associated steering angles
    images = np.empty([batch_size, HEIGHT, WIDTH, CHANNELS])
    steering = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 1.:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load(data_dir, center)
                # add the image and steering angle to the batch
                images[i] = transform(image)
                steering[i] = steering_angle
                i += 1
                if i == batch_size:
                    break
            return np.array(images).astype(np.float32), np.array(steering).astype(np.float32)
```

## Updated Function Caller:
```
X_train, y_train = aug_pipeline('data', X_train, y_train, True)
X_valid, y_valid = aug_pipeline('data', X_valid, y_valid, False)
```

## Updated Function to test:

```
def aug_pipeline(data_dir, image_paths, steering_angles, is_training):
    # Generate training image given image paths and associated steering angles
    # Create numpy array to store augmented images
    images = np.empty([image_paths.shape[0], HEIGHT, WIDTH, CHANNELS])
    steering = np.empty(images_paths.shape[0])
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # Random augmentation for training data
            if is_training and np.random.rand() < 1.:
                image, steering_angle = augment
```















