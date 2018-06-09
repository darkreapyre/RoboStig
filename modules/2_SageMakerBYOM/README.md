# Bring Your Own Model to SageMaker - NVIDIA Model
<details><summary><b>Note to self</b></summary><p>
Explain why the built-in image classification algorithm can't be used here.
</p></details>

[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
![Architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

## Training Functions
__blah blah blah__  

<details><summary><b>Note to self</b></summary><p>
Make sure to highlight the fact that the necessary libraries have already been configured in the model template file.
</p></details>


### Building the model: `build_model()` Function
__Blah blah blah__

```python
def build_model():
    """
    Create the NVidia Model.
    """
    net = None

    return net
```

<details><summary><b>Solution (Click to expand)</b></summary><p>
```
def build_model():
    """
    Create the NVidia Model.
    """
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Lambda(lambda x: x/127.5-1.0)) #Normalization
        net.add(gluon.nn.Conv2D(channels=24, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=36, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=48, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(1164))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(50))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(10))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(1))
    net.hybridize()
    return net
```
</p></details>

### Resizing the Input Shapes: `transform()` Function
__BLAH BLAH BLAH__  

```python
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
    data  = None
    label = None
    return data, label
```
<details><summary><b>Solution (Click to expand)</b></summary><p>

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
</p></details>

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
<details><summary><b>Solution (Click to expand)</b></summary><p>

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