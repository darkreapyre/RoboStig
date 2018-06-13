# View the Data
A full log of the driving observations is recorded in the `driving_log.csv` file. This file also includes the images captured from the *Left*, *Center* and *Right* camera angles from the vehicle. You will leverage this log, as well as the images, to train your model. But first, it's a good idea to view the contents of this file and derive some insights.

In the following code cells, we use the `pandas` library to create a tabular format (DataFrame) and display the first five rows of the `driving_log.csv` file.

<<CELL>>

# Describe the Data
Since the `.csv` file is now loaded as a DataFrame, we can leverage the `pandas` library to perform descriptive statistics, as the following code cell shows:

<<CELL>>

From the data, you will notice that there are $8036$ observations or driving "log" entries, which will be used to train your __RoboStig__ model. You can also see that there are $7$ features (or columns) to classify or categorize the observations.

By describing the summary statistics of the DataFrame, there are some interesting details that are further highlighted, namely:

1. Two thirds of the *steering* observations are $0.0$. This highlighters that the majority of the observations have a $0$ degree steering angle. In other words, the majority of the observations record the vehicle driving straight. 
2. Two thirds of the *speed* observations clock the vehicle driving at $30 Mph$, which coincides with the maximum speed value.
3. There is a correlation between the *speed* and *brake* observations since two thirds of the data also shows that 

# Visualize the Data

---