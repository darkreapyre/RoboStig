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