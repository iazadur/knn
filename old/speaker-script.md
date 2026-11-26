# Group Presentation Speaker Script (English Only)
**Topic:** Image Recognition: Classifying Images Based on Nearby Feature Representations  
**Course:** Artificial Intelligence  
**Course Teacher:** Ahsan Ullah, Assistant Professor, Dept. of CSE, World University of Bangladesh  
**Primary Reference:** `KNN.md`  
**Target Duration:** ~10 minutes total (~8 min 30 sec slides + ~1 min 30 sec live Google Colab demo)

## Group Responsibility Split

| Member | Slides | Live Colab Role | Target Time |
|---|---|---|---|
| **Md Azadur Rahman (ID: 4478)** | Slides 1–4 | Runs the live Google Colab notebook | ~2m 30s slides + ~1m 30s demo |
| **Rakib Hossain (ID: 4056)** | Slides 5–9 | Points out the manual NumPy distance table during demo | ~3m 10s |
| **Aduri Akter (ID: 4032)** | Slides 10–14 | Points out the accuracy and K-curve during demo | ~2m 50s |

---

## Opening Note (Before Slide 1)
All three members stand together at the front. Keep `index.html` open in fullscreen (`F`) and have `image-recognition-demo.ipynb` already open and pre-connected in a second browser tab on Google Colab so switching takes 2 seconds.

---

## Slide 1: Title — Image Recognition: Classifying Images Based on Nearby Feature Representations
- **Speaker:** Md Azadur Rahman (4478)
- **Main Objective:** Greet the teacher and classmates, introduce the three group members, and anchor our topic to Application #3 of the KNN lecture.
- **Approximate Speaking Time:** `25 seconds`
- **Exact Speaking Script:**
  > "Assalamu Alaikum, respected Sir and dear classmates. Welcome to our Artificial Intelligence group presentation on **Image Recognition: Classifying Images Based on Nearby Feature Representations**.
  >
  > I am **Md Azadur Rahman**, ID 4478, and presenting with me today are **Rakib Hossain**, ID 4056, and **Aduri Akter**, ID 4032. Our topic comes directly from Application number 3 of Sir's K-Nearest Neighbors lecture, so we will follow our class KNN methodology step by step and finish with a live Google Colab demonstration."
- **Simple Explanation:** We are taking our class KNN lecture and applying it directly to recognize images.
- **Analogy:** Applying one familiar tool—KNN—to a new kind of data: pictures.
- **Transition to Slide 2:**
  > "Let's quickly look at our learning objectives for today."

---

## Slide 2: Learning Objectives
- **Speaker:** Md Azadur Rahman (4478)
- **Main Objective:** State the five lecture-aligned outcomes clearly and briefly.
- **Approximate Speaking Time:** `30 seconds`
- **Exact Speaking Script:**
  > "Following our lecture's five learning objectives, by the end of our talk we will be able to:
  > 1. Explain how KNN works and how an image becomes a numerical feature representation.
  > 2. Calculate distances between images using the Euclidean distance formula.
  > 3. Apply KNN step by step to classify images—and also touch on KNN regression.
  > 4. Explain how choosing K and scaling features affect accuracy.
  > 5. And finally, run real handwritten digit recognition in Python and discuss KNN's advantages and limitations."
- **Simple Explanation:** Five checkpoints: concept, distance math, worked examples, tuning K and scaling, and live Python code.
- **Analogy:** Our five-stop roadmap for the next ten minutes.
- **Transition to Slide 3:**
  > "Let's begin with the intuition we already know from class: the fruit shop example."

---

## Slide 3: From Fruits to Images — What Is Image Recognition?
- **Speaker:** Md Azadur Rahman (4478)
- **Main Objective:** Bridge the teacher's fruit-shop analogy to image classification and clarify that KNN is one way—not the only way—to classify images.
- **Approximate Speaking Time:** `45 seconds`
- **Exact Speaking Script:**
  > "In our lecture, Sir explained that KNN is a supervised machine learning algorithm for both classification and regression, built on one simple idea: **similar data points tend to belong to the same class**. Remember the fruit shop example? When a new fruit arrives, we compare its weight and size against known apples and oranges. If most of the nearest fruits are apples, we classify it as an apple.
  >
  > Now replace the fruit with an **image**. Image recognition means identifying what an image shows—for instance, deciding which handwritten digit from 0 to 9 is in a picture. We describe each image using numbers, compare those numbers with labeled training images, and let the closest images decide. One important clarification: **KNN is one simple, transparent way to classify images—not every image recognition system uses KNN.**"
- **Simple Explanation:** Fruit traits become image numbers; nearest fruits become nearest labeled images.
- **Analogy:** Matching a mystery fruit by `(weight, size)` is identical in logic to matching a mystery digit by its pixel numbers.
- **Transition to Slide 4:**
  > "So how exactly does a computer turn a picture into a list of numbers? Let's look at feature representations."

---

## Slide 4: Images as Feature Representations
- **Speaker:** Md Azadur Rahman (4478)
- **Main Objective:** Introduce the bridge concept (*pixels → feature vector*) using the 3×3 digit example and preview the 8×8 digits dataset.
- **Approximate Speaking Time:** `50 seconds`
- **Exact Speaking Script:**
  > "A computer does not see curves or shapes—it sees a grid of small squares called **pixels**, where each pixel holds a brightness or ink number. A **feature representation** is simply the ordered list of numbers that describes an image.
  >
  > Look at these tiny 3-by-3 black-and-white digits where `1` means ink and `0` means blank. If we read the grid row by row, the digit `1` becomes the 9-number list `[0, 1, 0, 0, 1, 0, 0, 1, 0]`, and the digit `0` becomes `[1, 1, 1, 1, 0, 1, 1, 1, 1]`. Notice our new query image on the right—it's a slightly noisy `1` that differs from training `1` in just one single pixel!
  >
  > In our Colab demo, each real handwritten digit is an 8-by-8 grayscale grid with values from 0 to 16, giving us a **64-number feature vector**. Once every image is a list of numbers, every single step of Sir's KNN lecture works unchanged. Now I will hand over to **Rakib**, who will explain the terminology, steps, and distance math."
- **Simple Explanation:** Read the pixel squares row by row into one list of numbers—that list is the image's feature representation.
- **Analogy:** Writing a chessboard row by row into a single line so a computer can compare two boards cell by cell.
- **Transition to Slide 5 (Speaker Hand-off):**
  > "Over to you, Rakib."

---

## Slide 5: Important Terminology
- **Speaker:** Rakib Hossain (4056)
- **Main Objective:** Recap the teacher's six core terms plus *feature representation* and *majority voting* in image context.
- **Approximate Speaking Time:** `35 seconds`
- **Exact Speaking Script:**
  > "Thank you, Azad. Before we calculate distances, let's quickly review our key terms from the lecture in the context of images:
  > - **K** is the number of nearest neighbor images we inspect.
  > - A **Neighbor** is a nearby labeled image in our training dataset.
  > - **Distance** measures how different two image feature vectors are.
  > - **Classification** predicts a class label, like Digit 0 or 1, while **Regression** predicts a numerical value.
  > - The **Training dataset** holds our previously labeled images.
  > - And we add two image-specific terms: **Feature representation**—the list of numbers describing an image—and **Majority voting**, which picks the most frequent class among the K neighbors."
- **Simple Explanation:** These eight words are the entire vocabulary of KNN image recognition.
- **Analogy:** Learning the rules and pieces before playing a board game.
- **Transition to Slide 6:**
  > "Now let's put these terms into the six-step KNN process."

---

## Slide 6: How Does KNN Classify an Image?
- **Speaker:** Rakib Hossain (4056)
- **Main Objective:** Explain the teacher's 6-step KNN process and the K = 3 visual diagram (2 blue, 1 orange → blue).
- **Approximate Speaking Time:** `40 seconds`
- **Exact Speaking Script:**
  > "Here is the exact six-step process from our lecture:
  > 1. **Store** the labeled training images as feature vectors—notice that KNN requires no explicit model training.
  > 2. **Choose** a value of K.
  > 3. **Calculate** the distance between our new image and every stored training image.
  > 4. **Select** the K nearest neighbors with the smallest distances.
  > 5. For **classification**, take a **majority vote** among those K neighbors.
  > 6. Or for **regression**, take the **average** of their numerical values.
  >
  > Look at the diagram on the right from our lecture: around the new query point `?`, the three nearest neighbors are **two blue points and one orange point**. With K = 3, the majority vote predicts the **blue** class."
- **Simple Explanation:** Store images, measure distances, pick the K closest ones, and vote.
- **Analogy:** Asking your 3 closest neighbors in a new street which neighborhood you are in.
- **Transition to Slide 7:**
  > "How do we calculate that distance mathematically? Let's look at Euclidean distance."

---

## Slide 7: Distance Calculation (Euclidean Distance)
- **Speaker:** Rakib Hossain (4056)
- **Main Objective:** Present the 1D, 2D, and n-D Euclidean formulas and apply the n-D formula to the 3×3 pixel example.
- **Approximate Speaking Time:** `40 seconds`
- **Exact Speaking Script:**
  > "KNN commonly uses **Euclidean distance**. With one feature, it simplifies to the absolute difference `|x_new - x_training|`. With two features, it is Pythagoras' theorem. And for images with **n** features—whether 9 pixels or 64 pixels—we use the general formula: square each difference `(a_i - b_i)^2`, sum them up, and take the square root. Squaring stops positive and negative differences from cancelling out.
  >
  > Applying this directly to our 3-by-3 noisy query from Slide 4: the distance to training `'1'(a)` is `sqrt(1) = 1.00`, to `'1'(b)` is `sqrt(2) = 1.41`, and to `'0'(b)` is `sqrt(5) = 2.24`. With K = 3, two neighbors are Digit 1 and one is Digit 0, so KNN correctly predicts **Digit 1**."
- **Simple Explanation:** Subtract matching features, square, add, and take the square root—smaller distance means more visually similar.
- **Analogy:** Measuring straight-line ruler distance between two dots on graph paper.
- **Transition to Slide 8:**
  > "Let's walk through two short worked examples in the exact style of our class lecture."

---

## Slide 8: Worked Example 1 — One Image Feature
- **Speaker:** Rakib Hossain (4056)
- **Main Objective:** Solve the 1-feature dark-pixel example mirroring the lecture's study-hours example, including tie-breaking and the "not a guarantee" caveat.
- **Approximate Speaking Time:** `45 seconds`
- **Exact Speaking Script:**
  > "Just like Sir's study-hours example, suppose we classify an image as **Digit 0** or **Digit 1** using a single feature: the **number of dark pixels**. A thin `1` has fewer dark pixels (5 or 6), while a closed loop `0` has more (12 to 14).
  >
  > Now a new image arrives with **8 dark pixels**, and we set **K = 3**. Using `|8 - X|`, the distances are: Image A = 3, Image B = 2, C = 4, D = 5, and E = 6. The three nearest neighbors are **B, A, and C**—giving two votes for Digit 1 and one vote for Digit 0. So the **predicted class is Digit 1**.
  >
  > Two important reminders from our lecture: first, if two images had equal distance—for instance if the new image had 9 dark pixels—we would need an explicit **tie-breaking rule**. Second, this is a prediction based on training data, **not a guarantee**."
- **Simple Explanation:** Find absolute differences from 8, pick the 3 smallest, and let them vote.
- **Analogy:** Identical to predicting Pass/Fail from study hours, just counting dark pixels instead of hours.
- **Transition to Slide 9:**
  > "Now let's extend this to multiple image features."

---

## Slide 9: Worked Example 2 — Multiple Image Features
- **Speaker:** Rakib Hossain (4056)
- **Main Objective:** Solve the 3-feature digit example step by step and hand off smoothly to Aduri on choosing K.
- **Approximate Speaking Time:** `50 seconds`
- **Exact Speaking Script:**
  > "Here each image has **three features** on a 0-to-10 scale: **Center Ink**, **Loop Score**, and **Vertical Stroke**. Digit 0s have low center ink and high loop scores; Digit 1s have high center ink and strong vertical strokes.
  >
  > Our unseen query is **`(3, 6, 4)`** with **K = 3**. Computing Euclidean distance for each training image: Image B is closest at `sqrt(0 + 1 + 0) = 1.00` (Digit 0), Image A is second at `sqrt(1 + 4 + 1) = sqrt(6) ≈ 2.45` (Digit 0), and Image D is third at `sqrt(41) ≈ 6.40` (Digit 1). Images C and E are much farther away.
  >
  > Our three nearest neighbors are **B, A, and D**—two votes for Digit 0 versus one for Digit 1—so KNN predicts **Digit 0**. Now we chose K = 3 here, but why 3? **Aduri** will explain how choosing K and scaling features work."
- **Simple Explanation:** Same Euclidean formula across 3 traits; the two Digit 0s are much closer than any Digit 1.
- **Analogy:** Matching a photo on three scores—center ink, loop, and vertical line—instead of just one.
- **Transition to Slide 10 (Speaker Hand-off):**
  > "Over to you, Aduri."

---

## Slide 10: Choosing the Value of K
- **Speaker:** Aduri Akter (4032)
- **Main Objective:** Explain overfitting vs. underfitting, click K = 1, 3, 5 live on Example 2 to show K = 5 flipping the answer, and show real digits accuracy vs. K.
- **Approximate Speaking Time:** `45 seconds`
- **Exact Speaking Script:**
  > "Thank you, Rakib. Choosing **K** has a major impact on KNN performance. As Sir taught us: a **small K** is sensitive to noise and single unusual points, which can cause **overfitting**. A **large K** pulls in far-away neighbors, which can cause **underfitting**. In practice, we pick K using **validation data or cross-validation**, and for **binary** problems we prefer an **odd K** to reduce ties.
  >
  > *(Click `K = 1`, `K = 3`, and `K = 5` on the slide)* Watch our Example 2 live: K = 1 and K = 3 both predict **Digit 0**, but when I click **K = 5**, all five images vote—2 Digit 0s vs. 3 Digit 1s—and the prediction **flips to Digit 1**! We see the same underfitting on real handwritten digits below: accuracy stays at **98.44%** for K = 1 to 5, then drops steadily to **88.89%** at K = 101."
- **Simple Explanation:** Too small K trusts one noisy neighbor; too big K lets far-away classes outvote the local neighborhood.
- **Analogy:** Asking 1 stranger (risky) vs. 3–5 locals (balanced) vs. polling the entire city (too broad).
- **Transition to Slide 11:**
  > "Next, let's check another crucial factor in distance calculation: feature scaling."

---

## Slide 11: Feature Scaling for Image Features
- **Speaker:** Aduri Akter (4032)
- **Main Objective:** Apply the teacher's age-vs-income scaling lesson to mixed image features and share the honest finding about same-scale raw pixels.
- **Approximate Speaking Time:** `45 seconds`
- **Exact Speaking Script:**
  > "In our lecture, Sir showed how annual income in the thousands drowns out age in Euclidean distance. The exact same problem happens with images whenever features have **different numerical ranges**.
  >
  > Look at Example 4: **Total Ink** ranges from 0 to 1024, while **Symmetry** ranges from 0 to 1. Our new image `(530, 0.85)` is clearly symmetric like Digit 0. Yet without scaling, Total Ink dominates so badly that Digit 7 looks closer (`5.04` vs. `10.00`), giving the **wrong** answer! After normalizing each feature by its range, Digit 0 wins cleanly (`0.05` vs. `0.65`).
  >
  > One honest finding from our Colab test: when all 64 features are raw pixels already on the **same 0-to-16 scale**, raw pixels got **98.44%** while `StandardScaler` got **96.67%** because it amplified blank border pixels. So scale whenever feature ranges differ—not blindly."
- **Simple Explanation:** Put different-sized measurements onto a fair common scale before measuring Euclidean distance.
- **Analogy:** Measuring height in millimeters and weight in kilograms without converting units—millimeters would dominate.
- **Transition to Slide 12:**
  > "Before we show the Python code, let's briefly note how KNN also solves regression on images."

---

## Slide 12: KNN Regression with Images
- **Speaker:** Aduri Akter (4032)
- **Main Objective:** Connect directly to the teacher's KNN regression section using an illustrative image-to-age average calculation.
- **Approximate Speaking Time:** `35 seconds`
- **Exact Speaking Script:**
  > "Because our lecture covers both classification and regression, KNN can also predict a **numerical value** from an image by taking the **average** of its K nearest neighbors—just like Sir's house-price example.
  >
  > Suppose we estimate a person's **age** from a simple face texture score on a 0-to-10 scale. For a new face with score **3** and **K = 3**, the three closest training faces are B (age 25), C (age 35), and A (age 20). Taking their average—`(25 + 35 + 20) / 3`—gives a predicted age of **26.67 years**, matching `KNeighborsRegressor` in our notebook."
- **Simple Explanation:** Classification uses majority voting for a label; regression takes the arithmetic mean for a number.
- **Analogy:** Estimating a house price from 3 nearby houses is the same math as estimating an age from 3 similar face images.
- **Transition to Slide 13:**
  > "Now let's look at how clean this is to implement in Python."

---

## Slide 13: Python Implementation & Colab Benchmark
- **Speaker:** Aduri Akter (4032)
- **Main Objective:** Show the teacher-style scikit-learn code and report the verified `load_digits` test accuracy.
- **Approximate Speaking Time:** `40 seconds`
- **Exact Speaking Script:**
  > "Our Python code follows the exact structure from Sir's slides. We load Scikit-learn's `load_digits` dataset—1,797 real 8-by-8 handwritten digits—and split it into **1,347 training images** and **450 unseen test images**.
  >
  > Next we create `KNeighborsClassifier(n_neighbors=3)`, call `model.fit(X_train, y_train)` to store the training representations, and call `model.predict(X_test)`. On those 450 unseen images, KNN gets **443 right—98.44% accuracy!** And in our notebook, we also compute all 1,347 Euclidean distances manually with NumPy to verify that `scikit-learn` returns the exact same three neighbors."
- **Simple Explanation:** Four short steps in Python: split data, set K = 3, store training images, predict unseen images.
- **Analogy:** Scikit-learn automates the exact table calculations Rakib did by hand—just across 1,347 images in milliseconds.
- **Transition to Slide 14:**
  > "Let's wrap up with the advantages, limitations, and applications before Azad shows our live Colab notebook."

---

## Slide 14: Advantages, Limitations, Applications & Live Demo Transition
- **Speaker:** Aduri Akter (4032)
- **Main Objective:** Summarize the teacher's pros/cons table with image context, highlight Application #3, note optional future work briefly, and hand over to Azad for the live Colab demo.
- **Approximate Speaking Time:** `45 seconds`
- **Exact Speaking Script:**
  > "To summarize using our lecture's evaluation table:
  > - **Advantages:** KNN is simple and easy to understand, requires no explicit model training, handles both classification and regression, and captures nonlinear digit boundaries naturally.
  > - **Limitations:** Prediction becomes slow on large image collections because every new image is compared against all stored images; it is sensitive to feature scaling, the choice of K, and noisy or irrelevant pixels.
  >
  > Remember, **Image Recognition** is one of five major KNN applications alongside recommendations, healthcare, cybersecurity, and customer segmentation. Note that our 98.44% score is on small, clean, centered digits—for complex real-world photographs, modern systems first extract stronger features (such as with neural networks) as future work beyond our lecture.
  >
  > Now **Azad** will switch to Google Colab to run our live demonstration!"
- **Simple Explanation:** Simple and transparent on clean images, but slow and noise-sensitive on huge real-world photo sets.
- **Analogy:** A great microscope for small clean samples; huge messy collections need stronger feature extraction first.
- **Transition to Live Colab Demo (Speaker Hand-off):**
  > "Over to Azad for the live demo!"

---

## Live Google Colab Demo Guide (~1 minute 30 seconds — `image-recognition-demo.ipynb`)

*Keep the notebook already run (or run the key cells live) and scroll cleanly through 4 highlights:*

1. **Azad (35s — Sections 3 & 5):** Show the `Digit 0` 8×8 pixel grid right beside its 64-number feature vector bar chart. Then run the manual NumPy Step 3 cell for test image #0.
2. **Rakib (25s — Sections 5 & 6):** Point at the printed table of the 5 smallest Euclidean distances (`18.55, 20.10, 20.17` — all Digit 1), show the 3 nearest neighbor images side by side, and point out that `KNeighborsClassifier` matches the manual NumPy result identically.
3. **Aduri (25s — Sections 7, 8 & 9):** Point at the **`98.44%` (`443 / 450`)** test accuracy, the 7 misclassified digits with their 3 neighbors, and the **Accuracy vs. K** curve dropping to `88.89%` at K = 101.
4. **Azad (10s — Section 10, if time allows):** Show the custom hand-drawn `0` test cell predicting **Digit 0** with `[0, 0, 0]` as its 3 nearest neighbors (and mention that setting `UPLOAD_OWN_IMAGE = True` lets anyone upload their own photo).

---

## Closing Statement & Q&A Prep

**Closing (Azad, 10 seconds):**
> "That concludes our presentation and live demonstration. Thank you very much, Sir, and thank you everyone—we would be happy to answer any questions!"

**Quick Q&A Role Split:**
- **Azad:** Pixels, feature representation, dataset & Colab code.
- **Rakib:** Euclidean formula, tie-breaking, worked examples, lazy learning ("no explicit training").
- **Aduri:** Choosing K (overfit/underfit/odd K), feature scaling (and why raw pixels didn't need StandardScaler), regression, advantages/limitations.
