# Image Recognition: Classifying Images Based on Nearby Feature Representations
## Study Guide (Teacher-Aligned)

**Course:** Artificial Intelligence
**Course Teacher:** Ahsan Ullah, Assistant Professor, Dept. of CSE, World University of Bangladesh
**Primary reference:** Lecture notes *K-Nearest Neighbors (KNN)* (`KNN.md`)
**Group members:** Md Azadur Rahman (ID: 4478) · Rakib Hossain (ID: 4056) · Aduri Akter (ID: 4032)

---

## How to Use This Guide

Our topic comes straight from the teacher's lecture: it is item **#3 in the "Applications of KNN" list**:

> *"Image recognition: classifying images based on nearby feature representations."*

This guide therefore **follows the same order as the teacher's KNN lecture** and applies each idea to images. Every worked example mirrors one of the teacher's examples, and every number was checked in `image-recognition-demo.ipynb`.

| Guide section | Teacher's lecture section | Slide(s) |
|---|---|---|
| 1. Learning objectives | Learning Objectives | 2 |
| 2. Introduction: from fruits to images | Introduction to KNN | 3 |
| 3. Images as feature representations | *Bridge concept (added)* + Applications #3 | 4 |
| 4. Important terminology | Important Terminology | 5 |
| 5. How KNN classifies an image | How Does KNN Work? | 6 |
| 6. Distance calculation | Distance Calculation | 7 |
| 7. Worked Example 1 (one feature) | Worked Example: KNN Classification | 8 |
| 8. Worked Example 2 (multiple features) | KNN Regression – Multiple Features | 9 |
| 9. Choosing the value of K | Choosing the Value of K | 10 |
| 10. Feature scaling | Feature Scaling in KNN | 11 |
| 11. KNN regression with images | KNN Regression – Practical Example | 12 |
| 12. Python implementation and results | Python code sections | 13 + demo |
| 13. Advantages, limitations, applications | Advantages and Limitations; Applications | 14 |
| 14–17. Full walk-through, glossary, misunderstandings, Q&A | — | — |

---

## 1. Learning Objectives

Adapted from the teacher's five objectives. By the end, we should be able to:

1. Explain the concept and working principle of KNN, and how it classifies **images**.
2. Calculate distances between **image feature representations** using the Euclidean distance formula.
3. Apply KNN step by step to a simple **image classification** problem.
4. Explain the importance of selecting an appropriate value of **K** (and of feature scaling).
5. Identify the advantages, limitations and practical applications of KNN in image recognition.

---

## 2. Introduction: From Fruits to Images

### 2.1 What the teacher says about KNN
- KNN is a **supervised machine learning** algorithm used for **classification** and **regression**.
- **Basic idea:** data points with similar characteristics are likely to belong to the same class or to have similar values.
- **Key concept:** KNN predicts the class or value of a new data point by examining its **nearest neighbors in the training dataset**.

### 2.2 The teacher's fruit example
A new fruit arrives at a shop. To decide whether it is an **apple** or an **orange**, we compare its characteristics (**weight** and **size**) with fruits we have already identified. If most of the nearest fruits are apples, KNN classifies the new fruit as an apple.

### 2.3 The same idea for images

| Fruit example (lecture) | Image example (our topic) |
|---|---|
| A fruit | An image |
| Described by (weight, size) | Described by numbers: pixel values or image features |
| Known fruits with labels | Training images with labels (e.g., "Digit 0", "Digit 1") |
| Nearest fruits | Nearest images: images whose numbers are closest |
| Most neighbors are apples → apple | Most neighbor images are "Digit 1" → Digit 1 |

> **Key sentence:** *Once an image is written as a list of numbers, every step of the teacher's KNN lecture works without change.*

### 2.4 What is image recognition?
**Image recognition** is the ability of a computer to identify what an image shows. In this presentation we focus on **image classification**: giving one **class label** to a whole image, e.g., deciding which digit (0–9) a handwritten image shows.

> **Important:** KNN is **one** way to classify images. It is not the only way. Many other methods exist. KNN is useful because it is simple, transparent, and directly uses the idea of *nearby* feature representations.

---

## 3. Images as Feature Representations (the bridge concept)

The teacher's lecture does not define "feature representation", so we add this one bridge concept.

### 3.1 Pixels
A digital image is a grid of small squares called **pixels**. In a **grayscale** image each pixel holds one number for its brightness or ink.
- Our dataset (Scikit-learn's handwritten digits) uses **8 × 8 = 64 pixels** per image, each from **0 (white)** to **16 (dark ink)**.
- *(Colour images store three numbers per pixel, for red, green and blue. We do not need colour for this topic.)*

### 3.2 Feature representation
A **feature representation** is the **list of numbers that describes an image**. It plays the same role as (weight, size) for a fruit. There are two simple kinds:

1. **Raw pixel values:** read the pixel grid row by row into one long list (a **feature vector**).
2. **Image features:** a few measured properties, for example "number of dark pixels", "how much ink is in the center", or "does the digit have a closed loop?".

### 3.3 Example: 3 × 3 images become 9-number vectors

```text
 "1"            "0"            Query (a noisy "1")
 . # .          # # #          . # .
 . # .          # . #          . # .
 . # .          # # #          # # .

 # = ink = 1, . = blank = 0, read row by row:

 "1"   -> [0,1,0, 0,1,0, 0,1,0]
 "0"   -> [1,1,1, 1,0,1, 1,1,1]
 Query -> [0,1,0, 0,1,0, 1,1,0]
```

Each image is now **one point** described by 9 numbers (9 features). An 8 × 8 digit image is one point described by **64** numbers. Images that look alike have similar numbers, so they are **near** each other. That is exactly what "nearby feature representations" means.

---

## 4. Important Terminology

The teacher's six terms, plus two needed for images:

| Term | Explanation (lecture wording, applied to images) |
|---|---|
| **K** | Number of nearest neighbors considered |
| **Neighbor** | A nearby data point in the training dataset, here a **nearby training image** |
| **Distance** | A measure of how different or far apart two data points (images) are |
| **Classification** | Predicting a category or class (e.g., which digit) |
| **Regression** | Predicting a numerical value (e.g., a person's age from a face image) |
| **Training dataset** | Previously labeled data (labeled images) used to make predictions |
| **Feature representation** *(added)* | The list of numbers that describes an image (pixel values or image features) |
| **Majority voting** *(from the lecture's steps)* | Choosing the class that appears most often among the K neighbors |

---

## 5. How KNN Classifies an Image

### 5.1 The teacher's illustration
> The three nearest neighbors of the new point are **two blue points and one orange point**. With **K = 3**, the prediction is the **blue** class.

```text
            o   (orange)
     b
        b  [?]          K = 3 circle around "?"
     b          o       -> 2 blue + 1 orange -> Predicted: blue
          o
```

### 5.2 The KNN classification process (teacher's 6 steps, applied to images)
1. **Store the training data:** keep every labeled training image as a feature vector.
2. **Choose a value of K.**
3. **Calculate the distance** between the new image and each training image.
4. **Select the K nearest neighbors** (the K smallest distances).
5. **Classification:** use **majority voting** to decide the predicted class.
6. **Regression:** calculate the **average** (or another suitable aggregate) of the neighbors' values.

> KNN does **no explicit model training**. "Training" means storing the images; all the work happens when a new image must be predicted.

---

## 6. Distance Calculation

The teacher uses **Euclidean distance**.

**Two features**, points $A=(x_1,y_1)$ and $B=(x_2,y_2)$:
$$d(A,B)=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$$

**n features:**
$$d(A,B)=\sqrt{\sum_{i=1}^{n}(a_i-b_i)^2}$$
where $a_i$ and $b_i$ are the values of the $i$-th feature and $n$ is the number of features.

**One feature:** the formula becomes the absolute difference, $d=|x_{new}-x_{training}|$.

### Why square and then take the square root?
- **Squaring** stops negative and positive differences from cancelling each other out. Without it, $-2$ and $+2$ would add up to 0.
- The **square root** brings the result back to the original unit, just like Pythagoras' theorem.

### Applying it to the 3 × 3 pixel images (n = 9)
We add one more example of each digit:
- "1"(b) = [1,1,0, 0,1,0, 0,1,0], a "1" with a small top flag
- "0"(b) = [1,1,1, 1,0,1, 1,1,0], a "0" with a missing corner

| Training image | Squared differences summed | Distance | Class |
|---|---|---|---|
| "1"(a) | 1 | √1 = **1.00** | Digit 1 |
| "1"(b) | 2 | √2 ≈ **1.41** | Digit 1 |
| "0"(b) | 5 | √5 ≈ **2.24** | Digit 0 |
| "0"(a) | 6 | √6 ≈ **2.45** | Digit 0 |

With **K = 3** the neighbors are "1"(a), "1"(b) and "0"(b): **2 votes Digit 1 vs. 1 vote Digit 0, so Predicted class = Digit 1.**
The noisy query is correctly recognised as a "1".

---

## 7. Worked Example 1: One Image Feature
*(Mirrors the teacher's "study hours → Pass/Fail" example.)*

**Problem:** Classify a small handwritten digit image as **Digit 0** or **Digit 1** using **one feature: the number of dark pixels**. (A "1" is a thin stroke, so it has few dark pixels; a "0" is a loop, so it has more.)

### Training dataset

| Image | Dark pixels (X) | Class (Y) |
|---|---|---|
| A | 5 | Digit 1 |
| B | 6 | Digit 1 |
| C | 12 | Digit 0 |
| D | 13 | Digit 0 |
| E | 14 | Digit 0 |

A new image has **8 dark pixels**. Predict its class using KNN with **K = 3**.

### Step 1: Calculate the distances
Only one feature, so $d=|x_{new}-x_{training}|$.

| Image | Dark pixels | Distance from 8 | Class |
|---|---|---|---|
| A | 5 | 3 | Digit 1 |
| B | 6 | **2** | Digit 1 |
| C | 12 | 4 | Digit 0 |
| D | 13 | 5 | Digit 0 |
| E | 14 | 6 | Digit 0 |

### Step 2: Select the 3 nearest neighbors
B (2), A (3), C (4), with classes Digit 1, Digit 1, Digit 0.

### Step 3: Apply majority voting
Two neighbors are Digit 1, one is Digit 0.
$$\text{Predicted class} = \text{Digit 1}$$

> As in the lecture: this is a **prediction based on the available training data**, not a guarantee.

**About ties:** in the teacher's example, students A and D had equal distances, so a tie-breaking rule was needed ("assume A is selected"). Ties can happen with images too. For example, if the new image had **9** dark pixels, A and D would both be at distance 4. A tie-breaking rule must then be chosen. (Scikit-learn simply keeps the image that appears first in the training data.)

---

## 8. Worked Example 2: Multiple Image Features
*(Mirrors the teacher's multi-feature example and its step format.)*

**Problem:** Each image is described by **three features**, each on a 0–10 scale:
- $x_1$ = **Center Ink**: a "0" is empty in the middle (low); a "1" passes through the middle (high)
- $x_2$ = **Loop Score**: a "0" is a closed loop (high); a "1" is not (low)
- $x_3$ = **Vertical Stroke**: a "1" is a vertical line (high)

### Step 1: Store the training data

| Image | $x_1$ | $x_2$ | $x_3$ | Class |
|---|---|---|---|---|
| A | 2 | 8 | 3 | Digit 0 |
| B | 3 | 7 | 4 | Digit 0 |
| C | 8 | 2 | 9 | Digit 1 |
| D | 7 | 3 | 8 | Digit 1 |
| E | 9 | 1 | 9 | Digit 1 |

### Step 2: Choose the value of K
New image: $X_{new}=(3,6,4)$, with $K=3$.

### Step 3: Calculate the distances
$$d=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+(x_3-y_3)^2}$$

| Image | Calculation | Distance | Class |
|---|---|---|---|
| A | √((3−2)² + (6−8)² + (4−3)²) = √(1+4+1) = √6 | **2.45** | Digit 0 |
| B | √((3−3)² + (6−7)² + (4−4)²) = √(0+1+0) = √1 | **1.00** | Digit 0 |
| C | √((3−8)² + (6−2)² + (4−9)²) = √(25+16+25) = √66 | 8.12 | Digit 1 |
| D | √((3−7)² + (6−3)² + (4−8)²) = √(16+9+16) = √41 | **6.40** | Digit 1 |
| E | √((3−9)² + (6−1)² + (4−9)²) = √(36+25+25) = √86 | 9.27 | Digit 1 |

### Step 4: Select the K nearest neighbors

| Selected image | Distance | Class |
|---|---|---|
| B | 1.00 | Digit 0 |
| A | 2.45 | Digit 0 |
| D | 6.40 | Digit 1 |

### Step 5: Apply majority voting
2 × Digit 0 vs. 1 × Digit 1, so **Predicted class = Digit 0**.

---

## 9. Choosing the Value of K

From the lecture:
- **Small K:** sensitive to noise and to individual data points, so it may **overfit**.
- **Large K:** considers more neighbors and is less sensitive to noise, but may cause **underfitting**.
- **An appropriate K** balances these effects. It is generally chosen with **validation data or cross-validation**.
- For **binary** classification, an **odd K** is often used to reduce the chance of a voting tie.

### Trying different K on Worked Example 2 (as the teacher suggests, "try K = 1 and K = 5")

| K | Neighbors | Votes | Prediction |
|---|---|---|---|
| 1 | B | 1 × Digit 0 | **Digit 0** |
| 3 | B, A, D | 2 × Digit 0, 1 × Digit 1 | **Digit 0** |
| 5 | B, A, D, C, E | 2 × Digit 0, 3 × Digit 1 | **Digit 1** ← changed! |

With K = 5, **every** training image votes, including far-away ones, and the answer flips. This is underfitting in a tiny dataset.

### On real images (Colab, 450 test digits)

| K | 1 | 3 | 5 | 7 | 9 | 15 | 25 | 51 | 101 |
|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 98.44% | 98.44% | 98.44% | 98.00% | 97.56% | 96.89% | 95.78% | 92.44% | 88.89% |

Small K values work best here. Accuracy falls steadily as K grows very large (underfitting).
*Note: the digits problem has 10 classes, so an odd K does **not** guarantee no ties. The "odd K" rule is for binary problems.*

---

## 10. Feature Scaling for Image Features

The lecture explains that KNN relies on distances, so a feature with a **much larger numerical range** can dominate. The teacher's example is age (18–70) vs. income (10,000–100,000). The solution is **standardization or normalization**.

Standardization: $z=\dfrac{x-\mu}{\sigma}$. Normalization (by range): $x' = \dfrac{x}{\text{range}}$ (or $\dfrac{x-\min}{\max-\min}$).

### Worked Example 4: total ink vs. symmetry
Two image features: **total ink** (0–1024) and **symmetry** (0–1).

| Image | Total ink | Symmetry | Class |
|---|---|---|---|
| P | 520 | 0.90 | Digit 0 (round, symmetric) |
| Q | 525 | 0.20 | Digit 7 (not symmetric) |

New image: (530, 0.85), very symmetric like a "0". Nearest neighbor (K = 1):

| | d(new, P) | d(new, Q) | Nearest |
|---|---|---|---|
| **Without scaling** | √(10² + 0.05²) = 10.0001 | √(5² + 0.65²) = 5.0421 | Q → Digit 7 ✗ |
| **Normalized by range** (ink ÷ 1024) | 0.0509 | 0.6500 | P → Digit 0 ✓ |

Without scaling, total ink decides everything and symmetry is ignored.

### Important nuance for raw pixels
When **all** features are pixels on the **same** 0–16 scale, no feature dominates because of its units. In the Colab notebook:
- raw pixels: **98.44%**
- raw pixels + `StandardScaler`: **96.67%**

Here, standardizing gives almost-always-blank border pixels the same weight as the important center pixels. **Conclusion:** scale when features have **different ranges**; scaling is not automatically better.

---

## 11. KNN Regression with Images
*(Mirrors the teacher's house-price example.)*

KNN can also predict a **number**: take the **average** of the K neighbors' values.

**Illustrative problem:** estimate a person's **age** from one face-image feature, a *texture score* (0–10). The numbers are made up for teaching.

| Face image | Texture score | Age |
|---|---|---|
| A | 1 | 20 |
| B | 2 | 25 |
| C | 4 | 35 |
| D | 6 | 45 |
| E | 8 | 60 |

New face: texture score **3**, with **K = 3**.
- Distances: A = 2, B = 1, C = 1, D = 3, E = 5
- 3 nearest: B (25), C (35), A (20)

$$\text{Prediction}=\frac{25+35+20}{3}=\frac{80}{3}=26.67\text{ years}$$

Scikit-learn's `KNeighborsRegressor(n_neighbors=3)` gives the same **26.67**.

---

## 12. Python Implementation and Results

The code follows the teacher's pattern: training data → choose K → fit → predict.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Training data (each image = 64 pixel values)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=42, stratify=digits.target)

# Step 2: Choose K = 3
model = KNeighborsClassifier(n_neighbors=3)

# Step 3: "Train" (store the training images)
model.fit(X_train, y_train)

# Step 4: Predict new images
prediction = model.predict(X_test)
```

### Results from `image-recognition-demo.ipynb`
- **Dataset:** 1,797 handwritten digit images (8 × 8), split into **1,347 training** and **450 test** images.
- **Manual KNN for one test image** (NumPy only): 1,347 distances calculated. The 3 nearest training images have distances **18.55, 20.10, 20.17**, all **Digit 1**. Predicted class = Digit 1 (correct). Scikit-learn finds exactly the same neighbors.
- **All test images with K = 3:** **443 / 450 correct = 98.44%**. Only **7** mistakes, mostly unusually written digits whose nearest neighbors belong to another class.

> These images are small, clean and centered. The 98.44% result does **not** mean raw-pixel KNN would work this well on real-world photographs.

---

## 13. Advantages, Limitations and Applications

### Advantages and limitations (teacher's table, with image examples)

| Advantages | In image recognition |
|---|---|
| Simple and easy to understand | Prediction = "which stored images look most similar?" |
| No explicit model training is required | `fit()` only stores the labeled images |
| Can solve classification and regression problems | Digit class; age from a face image |
| Can model nonlinear decision boundaries | Separates 10 digit shapes without any formula |

| Limitations | In image recognition |
|---|---|
| Prediction can be slow for large datasets | Each new image is compared with **every** stored image |
| Sensitive to feature scaling | Total ink vs. symmetry example |
| Performance depends on the choice of K | Accuracy changes with K (98.44% → 88.89%) |
| Sensitive to irrelevant features and noisy data | Blank border pixels and messy handwriting cause mistakes |

### Applications of KNN (teacher's list)
1. **Recommendation systems:** finding similar users or items.
2. **Healthcare:** identifying similar patient records to support classification.
3. **Image recognition: classifying images based on nearby feature representations** ← *our topic* (e.g., handwritten digit recognition).
4. **Cybersecurity:** identifying network traffic patterns similar to known attacks.
5. **Customer segmentation:** grouping customers by similar characteristics.

### Optional future work (beyond the lecture)
For real photographs, raw pixels are not a good representation, because lighting, position and background change the pixel values. Modern systems first extract better features (for example with neural networks) and can still use nearest-neighbor search on those features. This is **outside** the scope of the lecture and is only mentioned as further reading.

---

## 14. Complete Walk-Through: How an Unknown Image Is Classified

```text
 Unknown image (e.g., a handwritten digit)
            |
            v
 [1] Feature representation: write the image as numbers
     (64 pixel values, or a few image features)
            |
            v
 [2] Training dataset: labeled images stored as feature vectors
     (KNN "training" = storing them)
            |
            v
 [3] Choose K (e.g., K = 3), using validation / cross-validation
            |
            v
 [4] Distance calculation: d = sqrt( sum (a_i - b_i)^2 )
     to EVERY training image (scale features first if their ranges differ)
            |
            v
 [5] Select the K nearest neighbors (smallest distances)
     (apply a tie-breaking rule if distances are equal)
            |
            v
 [6] Majority voting -> Predicted class
     (regression: average of neighbors' values)
```

---

## 15. Glossary

| Term | Meaning |
|---|---|
| Pixel | One small square of a digital image, holding a brightness/ink number |
| Grayscale image | An image with one number per pixel |
| Feature | One measurable number describing an image |
| Feature vector / feature representation | The ordered list of all features of one image |
| Euclidean distance | Straight-line distance $\sqrt{\sum (a_i-b_i)^2}$ |
| K | Number of nearest neighbors considered |
| Majority voting | Picking the most frequent class among the K neighbors |
| Overfitting | K too small: predictions follow noise or single unusual images |
| Underfitting | K too large: far-away images join the vote |
| Feature scaling | Standardization or normalization so that every feature counts fairly |
| Training dataset / test data | Labeled images stored by KNN / new images used to check predictions |
| Accuracy | Correct predictions ÷ total test images |

---

## 16. Common Misunderstandings

1. **"All image recognition uses KNN."** No. KNN is one simple method that classifies by nearby feature representations.
2. **"KNN learns a formula during training."** No. As the lecture says, no explicit model training is required; `fit()` stores the data.
3. **"A bigger K is always safer."** No. Large K causes underfitting (Worked Example 2 flips at K = 5; accuracy falls to 88.89% at K = 101).
4. **"Odd K always prevents ties."** Only for **binary** problems. With 10 digit classes, ties are still possible.
5. **"Scaling always improves KNN."** Scaling is needed when features have different ranges. For same-scale pixels it gave 96.67% vs. 98.44%.
6. **"98% on digits means KNN solves real-world image recognition."** No. The digits are tiny, clean and centered.
7. **"The prediction is certain."** No. It is a prediction based on the training data, not a guarantee.

---

## 17. Possible Audience Questions and Answers

**Q1. Why is KNN called a "lazy" learner?**
It does no explicit training. It stores the training images and does all the work (distance calculation and voting) only when a new image must be classified.

**Q2. How does a computer "see" an image?**
As a grid of numbers (pixels). We read the grid into a list of numbers, called the feature representation, and compare these lists with Euclidean distance.

**Q3. Why is K = 1 risky?**
A single unusual or mislabeled training image can decide the answer. That is overfitting. K = 3 or 5 asks several neighbors and reduces the effect of noise.

**Q4. What if two neighbors are at the same distance?**
A tie-breaking rule is needed, as in the teacher's example where student A was selected. Scikit-learn keeps the one that appears first in the training data.

**Q5. Why is KNN slow for large image collections?**
Every prediction compares the new image with every stored image: 1,347 comparisons in our demo, and millions in a large photo collection.

**Q6. Should we always scale image features?**
Scale when features have different ranges, such as total ink (0–1024) and symmetry (0–1). Raw pixels already share one range, and scaling did not help there.

**Q7. Can KNN be used for regression on images?**
Yes. Average the neighbors' values, e.g., estimate age from face-image features: (25 + 35 + 20) / 3 = 26.67.

**Q8. How did you choose K = 3?**
We compared accuracy for several K values. K = 1–5 gave the best result (98.44%). The lecture recommends validation data or cross-validation for this choice.
