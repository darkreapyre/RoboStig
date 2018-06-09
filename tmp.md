
<details><summary><b>Solution (Click to expand)</b></summary>
<p>
```
def transform(x, y):
    """
    Reshape the numpy arrays as 4D Tensors for MXNet.
    
    Arguments:
    x -- Numpy Array of input images
    y -- Numpy Array of labels
    
    Returns:
    x -- Numpy Array as (NCHW).
    y -- Label as Column vector.
    """
    data  = x.reshape(-1, 3, 66, 200)
    label = y.reshape(-1, 1)
    return data, label
```
</p>
</details>

### Loading and Transforming the Datasets: `load_data()` Function
__blah blah blah__

```python
def load_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Resized training and testing data along with training and testing labels.
    """

    return None
```

<details><summary><b>Solution (Click to expand)</b></summary>
<p>
```
def load_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Pre-processed training and testing data along with training and testing labels.
    """
    train_x = np.load(f_path+'/train_X.npy')
    train_y = np.load(f_path+'/train_Y.npy')
    train_X, train_Y = transform(train_x, train_y)
    test_x = np.load(f_path+'/valid_X.npy')
    test_y = np.load(f_path+'/valid_Y.npy')
    test_X, test_Y = transform(test_x, test_y)
    return train_X, train_Y, test_X, test_Y
```
</p></details>

### Measuring the Training vs. Validation Accuracy: `accuracy()` Function
__blah blah blah__