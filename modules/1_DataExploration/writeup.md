__Note to Self:__ *CODE CELLS* should change the formatting to have the line approximation before `###`.

__Note to Self:__ *Data Preprocessing* section has incorrect grammar in the second sentence. basically there should be an `and` instead of a `,`.

__Note to Self:__ *Image Transformations*
- First bullet has incorrect gramar.
- Section needs the objective for each bullet point in BOLD. Also the first sentence doesn't make sense.
- Add the following to the main sentence:

Along with extracting the necessary features from the data, a similar pre-processing steps need to be applied to the images images themselves. This is necessary to align with the image formatting requirements of the NVIVIA model.

---

---
## Image Augmentation

<<MARKDOWN CELL>>
As already mentioned, you will need to add more diversity to the Training data, so the data isn;t skewed toward only driving straight. In order to accomplish this, you will take some of the existing images (primarily the __left__ and __right__ camera images) and adjust them just enough so that they can add the additional variance required.

<div class="alert alert-primary" role="alert">
<strong>Info: </stong>Many of the Python Machine Learning framework have built in methods to augment images. You can leverasge these if you wish, <strong>BUT</strong> make sure to adjust the steering angle data accordingly.
</div>

__Exercise:__ For the next exercise, you will implement *some* of the image augmentation techniques.

1. __Randomly flip the images.__ Here you will want to take $50%$ of all images and randomly flip them vertically. If the image is flipped, then the reverse of the steering angle must be applied.
<div class="alert alert-success" role="alert">
<strong>Hint: </stong>The `cv2.flip()` function may be helpful.
</div>

2. __Randomly shift the images.__ Here the image is shifted vertically and horizontally by a $0.002^O$ pixel shift. The steering angle is adjusted accordingly.
<div class="alert alert-primary" role="alert">
<strong>Info: </stong>This function has already been implemented for you.
</div>

3. __Randomly distort the images.__ This augmentation technique applies a random distortion to the the images and also randomly adjusts the brightness of the images.
<div class="alert alert-primary" role="alert">
<strong>Info: </stong>This function has already been implemented for you.
</div>

4. __Randomly adjust the brightness of the images.__ Here you will implement a function adjust the brightness, either making the image brighter or lowering the brightness (making the image darker).
<div class="alert alert-success" role="alert">
<strong>Hint: </stong>The `cv2.cvtColor()` function may be helpful, paying attention to HSV (Hue, Saturation, Value) or HSB.
</div>

<</MARKDOWN CELL>>


<<CODE CELL>>

# Image Augmentation: Random Flip
def random_flip(image, steering_angle):
    """
    Randomly - 50% of the time - flip the image left from left to right and vice-versa. 
    Additionally, adjust the steering angle accordingly.

    Arguments:
    image -- pre-processed input image.
    steering_amngle -- pre-processed steering angle.

    Returns:
    image -- flipped image.
    steering_angle -- adjusted steering angle.
    """
    ###   START OF YOUR CODE (≈3 lines of code)   ###

    return None
    ###   END OF YOUR CODE ###

<</CODE CELL>>

<<MARKDOWN CELL>>

<details><summary><b>Solution (Click to expand)</b></summary><p>

```
# Image Augmentation: Random Flip
def random_flip(image, steering_angle):
    """
    Randomly flip the 50% of the images from left to right and vice-versa.
    Additionally, adjust the steering angle accordingly.

    Arguments:
    image -- pre-processed input image.
    steering_amngle -- pre-processed steering angle.

    Returns:
    image -- flipped image.
    steering_angle -- adjusted steering angle.
    """
    # Randomly select 0.5 of the images
    if np.random.rand() < 0.5:
        # Apply the flip function to the vertical axis.
        image = cv2.flip(image, 1)

        # Adjust the steering angle to the reverse of the current steering angle.
        steering_angle = -steering_angle

    # Return the "flipped" image and new steering angle.

    return image, steering_angle
```

</p><details>

<</MARKDOWN CELL>>

<<CODE CELL>>

# Image Augmentation: Random Translate
def translate(image, steering_angle, x_range, y_range):
    """
    Randomly shift (translate) the image vertically and horizontally.

    Arguments:
    image -- pre-processed input image.
    steering_angle -- pre-processed steering angle.
    x_range -- x-axis pixels.
    y_range -- y-axis pixels.

    Returns:
    image -- translated image.
    steering_angle -- adjusted steeing angle.
    """
    # Randomly adjust the x and y axis
    x_transform = x_range * (np.random.rand() - 0.5)
    y_transform = y_range * (np.random.rand() - 0.5)

    # Adjust the steering angle
    steering_angle += x_transform * 0.002
    m_transform = np.float32([[1, 0, x_transform], [0, 1, y_transform]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, m_transform, (width, height))

    return image, steering_angle

# Image Augmentation: Random Distortion
def distort(image):
    """
    Add distortion to random images and adjust the brightness.

    Arguments:
    image -- pre-processed input image.

    Returns:
    new_image -- distorted image.
    """
    # Create placeholder numpy array for the new image
    new_img = image.astype(float)

    # Add random brightness
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:, :, 0] + value) > 255
    if value <= 0:
        mask = (new_img[:, :, 0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)

    # Add random shadow 
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0, w)
    factor = np.random.uniform(0.6, 0.8)
    if np.random.rand() > .5:
        new_img[:, 0:mid, 0] *= factor
    else:
        new_img[:, mid:w, 0] *= factor
    
    # Randomly shift the horizon
    h, w, _ = new_img.shape
    horizon = 2 * h / 5
    v_shift = np.random.randint(-h / 8, h / 8)
    pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    pts2 = np.float32([[0, horizon + v_shift], [w, horizon + v_shift], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return new_img.astype(np.uint8)

# Image Augmentation: Random Brightness
def brightness(image):
    """
    Randomly adjust brightness of the image.

    Arguments:
    image -- pro-processed input image.

    Returns:
    HSV/HSB converted image.
    """
    ###   START OF YOUR CODE (≈4 lines of code)   ###
    
    return None
    ###   END OF YOUR CODE ###

<</CODE CELL>>

<<MARKDOWN CELL>>

<details><summary><b>Solution (Click to expand)</b></summary><p>

```
# Image Augmentation: Random Brightness
def brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Randomly adjust the brightness ratio and apply it
    # to the image
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio

    # Convert back to RGB and return the image
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```

</p>
</details>

<</MARKDOW  CELL>>

---

---

## Augmented Image Examples

```text
|--data
|  |--driving_log.csv
|  |--IMG
|  |  |--center_2016_12_01_13_30_48_287.jpg
|  |  |--left_2016_12_01_13_30_48_287.jpg
|  |  ...
```

---

---

## Image Augmentation Pipeline

1. Randomly choose right, left or center images.