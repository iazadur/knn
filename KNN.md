# K-Nearest Neighbors (KNN)

**Prepared by:** Ahsan Ullah
Assistant Professor, Dept. of CSE
World University of Bangladesh

---

## Learning Objectives

By the end of this lecture, students will be able to:

- Explain the concept and working principle of KNN.
- Calculate distances between data points using the Euclidean distance formula.
- Apply KNN to solve a simple classification problem.
- Explain the importance of selecting an appropriate value of K.
- Identify the advantages, limitations, and practical applications of KNN.

---

## Introduction to KNN

- K-Nearest Neighbors (KNN) is a **supervised machine learning** algorithm used for **classification** and **regression**.
- Basic idea: data points with similar characteristics are likely to belong to the same class or have similar values.
- Example: a new fruit is brought to a fruit shop. We want to identify whether it is an apple or an orange. We compare its characteristics (e.g., weight and size) with previously identified fruits.
- If most of the nearest fruits are apples, KNN classifies the new fruit as an apple.
- **Key concept:** KNN predicts the class or value of a new data point by examining its nearest neighbors in the training dataset.

---

## Important Terminology

| Term | Explanation |
|---|---|
| K | Number of nearest neighbors considered |
| Neighbor | A nearby data point in the training dataset |
| Distance | A measure of how different or far apart two data points are |
| Classification | Predicting a category or class |
| Regression | Predicting a numerical value |
| Training dataset | Previously labeled data used to make predictions |

---

## How Does KNN Work?

*Illustration:* The three nearest neighbors of the new point are two blue points and one orange point. With K = 3, the prediction is the **blue** class.

### KNN Classification Process

1. Store the training data with known class labels or numerical values.
2. Choose a value of K.
3. Calculate the distance between the new data point and each training data point.
4. Select the K nearest neighbors.
5. **Classification:** use majority voting to determine the predicted class.
6. **Regression:** calculate the average (or another suitable aggregate) of the neighbors' values.

---

## Distance Calculation

KNN commonly uses **Euclidean distance**.

For two points $A = (x_1, y_1)$ and $B = (x_2, y_2)$:

$$d(A, B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

For data with multiple features:

$$d(A, B) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

Where:

- $a_i$ and $b_i$ are the values of the $i$-th feature.
- $n$ is the number of features.

---

## Worked Example: KNN Classification

Dataset of students' study hours and exam results. Predict whether a new student will **Pass** or **Fail**.

### Training dataset

| Student | Study hours (X) | Result (Y) |
|---|---|---|
| A | 1 | Fail |
| B | 2 | Fail |
| C | 3 | Pass |
| D | 4 | Pass |
| E | 5 | Pass |

A new student studies for **2.5 hours**. Predict the result using KNN with **K = 3**.

### Step 1: Calculate the distances

Only one feature, so distance is the absolute difference:

$$d = |x_{new} - x_{training}|$$

| Student | Study hours | Distance from 2.5 | Result |
|---|---|---|---|
| A | 1 | 1.5 | Fail |
| B | 2 | 0.5 | Fail |
| C | 3 | 0.5 | Pass |
| D | 4 | 1.5 | Pass |
| E | 5 | 2.5 | Pass |

### Step 2: Select the 3 nearest neighbors

- The three nearest students are B, C, and A or D. Since A and D have equal distances, a tie-breaking rule is needed. For this example, assume **A** is selected.
- Their results:
  - Student B: Fail
  - Student C: Pass
  - Student A: Fail

### Step 3: Apply majority voting

Two students are labeled Fail, one is labeled Pass.

$$\text{Predicted class} = \text{Fail}$$

Therefore, the KNN model predicts that the new student will **Fail**. This is a prediction based on the available training data, not a guarantee of the actual exam result.

---

## KNN Regression – Practical Example

### Example: Predicting House Prices Using KNN Regression

**Problem:** A real estate company wants to predict the price of a new house based on its area (square feet). The company has data about five previously sold houses.

### Step 1: Store the training data

| House | Area (sq. ft.) | Price (Lakh BDT) |
|---|---|---|
| A | 1000 | 30 |
| B | 1200 | 35 |
| C | 1500 | 40 |
| D | 1800 | 50 |
| E | 2000 | 55 |

- Input feature (X) = House area
- Target value (Y) = House price
- This is a **regression** problem because we want to predict a numerical value.

### Step 2: Choose the value of K

A new customer wants to buy a house with an area of **1600 sq. ft.**

$$K = 3$$

The algorithm will select the three nearest houses from the training dataset.

### Step 3: Calculate the distance

Only one feature (house area):

$$Distance = |X_{new} - X_{training}|$$

New house area = 1600 sq. ft.

| House | Area | Price | Distance |
|---|---|---|---|
| A | 1000 | 30 | 600 |
| B | 1200 | 35 | 400 |
| C | 1500 | 40 | 100 |
| D | 1800 | 50 | 200 |
| E | 2000 | 55 | 400 |

For example:

- House A: $|1600 - 1000| = 600$
- House C: $|1600 - 1500| = 100$

### Step 4: Select the K nearest neighbors

K = 3, so we select the three houses with the smallest distances.

| Selected House | Area | Price | Distance |
|---|---|---|---|
| C | 1500 | 40 | 100 |
| D | 1800 | 50 | 200 |
| B | 1200 | 35 | 400 |

Notice that House B and House E both have a distance of 400. In this example, House B is chosen to break the tie.

### Step 5: Calculate the average price

In KNN regression, we calculate the average target value of the selected neighbors.

$$Prediction = \frac{Y_1 + Y_2 + Y_3}{K} = \frac{40 + 50 + 35}{3} = \frac{125}{3} = 41.67 \text{ Lakh BDT}$$

**Final prediction: 41.67 Lakh BDT**

### Step 6: Understand the process visually

The three selected neighbors have prices of 40, 50, and 35 lakh BDT. Their average gives the predicted house price.

*(Illustration of the arithmetic mean: values 4, 5, 6 → $\bar{x} = \frac{4+5+6}{3} = \frac{15}{3} = 5$, where $\bar{x}$ is the arithmetic mean / simple average.)*

### Python code (single feature)

```python
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Step 1: Training data
X = np.array([[1000],
              [1200],
              [1500],
              [1800],
              [2000]])
y = np.array([30, 35, 40, 50, 55])

# Step 2: Choose K = 3
model = KNeighborsRegressor(n_neighbors=3)

# Step 3: Train the model
model.fit(X, y)

# Step 4: Predict the price
new_house = np.array([[1600]])
prediction = model.predict(new_house)

# Step 5: Display result
print("Predicted House Price:", prediction[0], "Lakh BDT")
```

---

## KNN Regression – Example Using Multiple Features

### Problem: Predicting House Prices

Predict the price of a house based on:

1. Area (square feet)
2. Number of bedrooms
3. Number of bathrooms
4. Distance from the city centre (km)

### Step 1: Store the training data

| House | Area (sq. ft.) | Bedrooms | Bathrooms | Distance (km) | Price (Lakh BDT) |
|---|---|---|---|---|---|
| A | 1000 | 2 | 1 | 10 | 30 |
| B | 1200 | 2 | 2 | 8 | 35 |
| C | 1500 | 3 | 2 | 5 | 45 |
| D | 1800 | 3 | 3 | 4 | 55 |
| E | 2000 | 4 | 3 | 3 | 60 |

- Input features (X) = Area, Bedrooms, Bathrooms, Distance
- Target (Y) = House Price
- This is a regression problem because the output is a numerical value.

### Step 2: Choose the value of K

New house (test data):

| Area | Bedrooms | Bathrooms | City distance |
|---|---|---|---|
| 1600 sq. ft. | 3 | 2 | 5 km |

$$K = 3$$

### Step 3: Calculate the distance

Since there are four features, we use the Euclidean distance formula:

$$d = \sqrt{\sum_{i=1}^{n}(X_i - Y_i)^2}$$

For four features:

$$d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2 + (x_4 - y_4)^2}$$

Here, $X$ represents the new house and $Y$ represents an existing house.

**Distance for House A**

- New house: $(1600, 3, 2, 5)$
- House A: $(1000, 2, 1, 10)$

$$d_A = \sqrt{(1600-1000)^2 + (3-2)^2 + (2-1)^2 + (5-10)^2}$$
$$= \sqrt{360000 + 1 + 1 + 25} = \sqrt{360027} = 600.02$$

**Distances for all five houses**

| House | Euclidean Distance |
|---|---|
| A | 600.02 |
| B | 400.01 |
| C | 100.00 |
| D | 200.01 |
| E | 400.01 |

> **Note:** These are illustrative raw Euclidean distances. The area feature dominates because its numerical scale is much larger than the other features.

### Step 4: Select the K nearest neighbors

| Selected House | Distance | Price |
|---|---|---|
| C | 100.00 | 45 |
| D | 200.01 | 55 |
| B | 400.01 | 35 |

For the tie between B and E, B is selected in this example.

### Step 5: Calculate the average price

$$Prediction = \frac{45 + 55 + 35}{3} = \frac{135}{3} = 45 \text{ Lakh BDT}$$

**Predicted house price: 45 Lakh BDT**

### Python code (multiple features, with scaling)

```python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Step 1: Training data
# Features: Area, Bedrooms, Bathrooms, Distance
X = np.array([
    [1000, 2, 1, 10],
    [1200, 2, 2, 8],
    [1500, 3, 2, 5],
    [1800, 3, 3, 4],
    [2000, 4, 3, 3]
])

# Target: House price in lakh BDT
y = np.array([30, 35, 45, 55, 60])

# Step 2: Create KNN Regression model
# Standardize features because their scales differ
model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=3)
)

# Step 3: Train the model
model.fit(X, y)

# Step 4: New house information
new_house = np.array([[1600, 3, 2, 5]])

# Step 5: Predict house price
prediction = model.predict(new_house)
print("Predicted House Price:", prediction[0], "Lakh BDT")
```

### KNN Regression Map (5 Neighbours)

*(Figure: a contour map of predicted price (€) against Number of Rooms (x-axis) and Distance to Centre in km (y-axis), with a "New Property" marked by a **?**. Colours range from purple (~200,000) to yellow (~500,000+).)*

---

## Choosing the Value of K

The value of K has a significant impact on KNN performance.

- **Small K:** sensitive to noise and individual data points; may **overfit**.
- **Large K:** considers more neighbors, less sensitive to noise, but may cause **underfitting**.
- **An appropriate K:** balances these effects and is generally selected using validation data or cross-validation.

For binary classification, an **odd** value of K is often used to reduce the chance of a voting tie.

### Try different values of K

Same example: new student's study time is 2.5 hours.

- **K = 3** → Two Fail and one Pass among the three nearest neighbors → Predicted result: **Fail**
- (Try K = 1 and K = 5 as well.)

---

## Feature Scaling in KNN

Feature scaling is important because KNN relies on distance calculations.

Suppose we want to classify a person based on age and annual income:

- Age ranges from 18 to 70.
- Annual income ranges from 10,000 to 100,000.

The income feature has a much larger numerical range and may dominate the distance calculation. To avoid this, use **standardization** or **normalization** before applying KNN.

---

## Advantages and Limitations

| Advantages | Limitations |
|---|---|
| Simple and easy to understand | Prediction can be slow for large datasets |
| No explicit model training is required | Sensitive to feature scaling |
| Can solve classification and regression problems | Performance depends on the choice of K |
| Can model nonlinear decision boundaries | Sensitive to irrelevant features and noisy data |

---

## Applications of KNN

1. **Recommendation systems:** finding similar users or items.
2. **Healthcare:** identifying similar patient records to support classification.
3. **Image recognition:** classifying images based on nearby feature representations.
4. **Cybersecurity:** identifying network traffic patterns similar to known attacks.
5. **Customer segmentation:** grouping customers based on similar characteristics.

---

**Thank you**
