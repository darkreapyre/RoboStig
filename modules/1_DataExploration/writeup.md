## Image Augmentation
- Randomly choose right, left or center images.
- For left image, steering angle is adjusted by +0.2
- For right image, steering angle is adjusted by -0.2
- Randomly flip image left <--> right
- Randomly translate image horizontally with steering angle adjustment (0.002 per pixel shift)
- Randomly translate image virtically
- Randomly added shadows
- Randomly altering image brightness (lighter or darker)
- Randonly apply distortions

<CODE CELL>
# Randomly Flip image Left/Right
def random_flip(image, steering_angle):
    """
    Randomly flit the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

</CODE CELL>

<MARKDOWN CELL>

__BLAH BLAH BLAH --> SOLUTION__

</MARKDOWN CELL>

__ETC.......__

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