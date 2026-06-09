# Academic-Success-Predictor-ML

The goal of this project is to identify, at an early stage, whether there is a risk of a student becoming a dropout based on general demographic and socioeconomic characteristics of their background.

For this purpose, a supervised Machine Learning (ML) model was used.

---

## Dataset

The chosen dataset is the following:

**[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)**

It contains information from **4,424 students** and **36 variables**, including academic, demographic, and socioeconomic factors related to academic performance and school dropout.

The dataset was already preprocessed to eliminate missing values, and all categorical variables, except for the target attribute, were already encoded as numerical values.

The dataset originally contains the following attributes:

| Variable | Description |
|-----------|-------------|
| Marital status | Marital status |
| Application mode | Admission modality |
| Application order | Degree preference order |
| Course | Degree program enrolled in |
| Daytime/evening attendance | Whether the student studies during daytime or evening hours |
| Previous qualification | Previous academic qualification |
| Previous qualification (grade) | Previous qualification grade |
| Nationality | Nationality |
| Mother's qualification | Mother's educational level |
| Father's qualification | Father's educational level |
| Mother's occupation | Mother's occupation |
| Father's occupation | Father's occupation |
| Admission grade | Admission grade |
| Displaced | Displaced student |
| Educational special needs | Special educational needs |
| Debtor | Has financial debt with the institution |
| Tuition fees up to date | Tuition payments up to date |
| Gender | Gender |
| Scholarship holder | Has a scholarship |
| Age at enrollment | Age at enrollment |
| International | International student |
| Curricular units 1st sem (credited) | Courses credited by equivalency |
| Curricular units 1st sem (enrolled) | Courses enrolled in |
| Curricular units 1st sem (evaluations) | Evaluations completed |
| Curricular units 1st sem (approved) | Courses passed |
| Curricular units 1st sem (grade) | Semester average grade |
| Curricular units 1st sem (without evaluations) | Courses without evaluation |
| Curricular units 2nd sem (credited) | Courses credited |
| Curricular units 2nd sem (enrolled) | Courses enrolled in |
| Curricular units 2nd sem (evaluations) | Evaluations completed |
| Curricular units 2nd sem (approved) | Courses passed |
| Curricular units 2nd sem (grade) | Average grade |
| Curricular units 2nd sem (without evaluations) | Courses without evaluation |
| Unemployment rate | Unemployment rate |
| Inflation rate | Inflation rate |
| GDP | Gross Domestic Product |
| **Target** | **Dropout, Enrolled, Graduate** |

---

## Data Analysis

The most important step in building a model that works as expected is understanding the data being used. To achieve this, several exploratory analyses were performed, including reviewing sample instances and verifying that the dataset did not contain missing values.

Afterwards, the categorical features were analyzed to understand the imbalances present among the different categories.

<p align="center">
  <img src="categorical features barchars.png" alt="Categorical Features Distribution" width="90%" />
  <br>
  <em>Figure 1. Distribution of categorical features across the dataset.</em>
</p>

The distributions of the numerical features were also analyzed to identify where the majority of the data was concentrated.

<p align="center">
  <img src="continuous features histograms.png" alt="Numerical Features Distribution" width="90%" />
  <br>
  <em>Figure 2. Distribution of numerical features across the dataset.</em>
</p>

From both analyses, it can be clearly observed that most instances are concentrated around common categories or specific numerical ranges. This indicates that the dataset is highly imbalanced, which, according to the literature, is a common challenge when working with educational datasets.

Lastly, a Correlation Matrix was generated to identify variables that could potentially be redundant. However, it is important to note that since all attributes are already encoded, the Correlation Matrix can be misleading if interpreted incorrectly.

<p align="center">
  <img src="simplified correlation matrix.png" alt="Simplified Correlation Matrix" width="90%" />
  <br>
  <em>Figure 3. Simplified correlation matrix showing relationships between variables.</em>
</p>

---

## Preprocessing

As stated before, the goal of this project is to build a model capable of making an early prediction of whether a student is at risk of dropping out, based on general conditions available at the beginning of their academic journey.

To support this objective, several attributes were removed. These included:

* Attributes containing information that would not be available at the student's starting point.
* Attributes that were too specific to the institution where the data was collected.
* Macroeconomic factors related to the student's country of origin.

Additionally, the **Nationality** attribute was removed due to its high correlation with the **International** attribute. This decision was made because the **International** attribute already captures the potential challenges a student may face while studying in a country other than their own. Furthermore, **Nationality** presented a highly biased distribution and was therefore considered more likely to introduce noise than meaningful information.

Any instances belonging to the **Enrolled** category were also removed, as they did not represent a definitive outcome and therefore could not contribute to a clear classification objective.

After removing the Enrolled class, the dataset contained 3,630 instances:
- Graduate: 2,209
- Dropout: 1,421

Finally, the target variable was encoded into numerical labels. Since only two target categories remained after preprocessing, it was not necessary to use One-Hot Encoding. Instead, the labels were manually encoded.

### Feature Scaling

Machine Learning Models are sensitive to feature magnitudes, this is the reason scaling is used, to preserve the importance of the changes, without biasing the model towards big numbers. 

But scaling should only be done on continuous features where order implies something, as on other cases, it alters the information the model should have received.

Because of this, scaling was applied only on the isolated continuous features, and afterwards, they were concatenated back together.

---

## Train-Test Split

The dataset was divided into **80% training data** and **20% testing data**. The `stratify` parameter was used during the split to preserve the original class distribution of the target variable in both subsets, ensuring that the proportion of each class remained consistent across the training and testing sets.

---

## Evaluation Metrics

As educational datasets have a tendency to be imbalanced, accuracy alone can be misleading if the model performs better on the majority class. For this reason, we evaluated our model using **Precision**, **Recall**, **F1-Score**, and **AUC**, following the type of metrics used in related literature (Hashim et al., 2020; Wongvorachan et al., 2023; Bujang et al., 2021).

- **Precision:** Measures how many students predicted as Graduate or Dropout actually belonged to that class. In our case, it indicates how reliable the model is when predicting that a student is at risk of dropping out.
- **Recall:** Measures how many of the actual students were correctly identified as Graduate or Dropout by the model. For us, it shows how many real dropout cases were successfully detected.
- **F1-Score:** Combines Precision and Recall into a single metric, providing a general evaluation of the model's performance.
- **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between both classes across different classification thresholds. The higher the value, the better the model is at separating Graduate from Dropout cases.

Other metrics such as **TP Rate**, **FP Rate**, and **Accuracy** were also considered. However, they were not included in the final evaluation because they either overlap with information already provided by the selected metrics or are less informative when dealing with imbalanced datasets.

## Logistic Regression Model (Baseline Model)

[Google Colab Notebook](https://colab.research.google.com/drive/1rJd0RU-_aptsimCJQfmdmfpguI1dz40Z?usp=sharing)

A Logistic Regression model was used as the baseline model. This decision was based on the study by Hashim et al. (2020), where several supervised machine learning algorithms were compared for student performance prediction using demographic, academic background, and behavioral features. Their results showed that Logistic Regression achieved the best performance among the evaluated algorithms.

### Architecture

The model was implemented using Keras. It consists of a single output layer and contains no hidden layers. This means that the input features are directly connected to the output neuron, which returns the probability of a student belonging to either the **Dropout** or **Graduate** class.

### Hyperparameters

- **Epochs:** 50
- **Batch Size:** 32
- **Threshold:** 0.35
- **Activation Function:** Sigmoid
- **Loss Function:** Binary Cross-Entropy

### Evaluation

The model was evaluated with a threshold of **0.35**. The reason for selecting a relatively low threshold is to identify as many students at risk of dropping out as possible, as failing to take the necessary measures could negatively impact their academic careers and future opportunities. The tradeoff is a higher likelihood of generating false alarms and, consequently, misusing institutional resources.

Across the training history of the model, we can observe that the accuracy and loss of the validation were hand-in-hand with the one of the training, sometimes oscillating around it. This means the model doesn't show signs of overfitting or underfitting.

Another important observation is that accuracy increases quickly and stabilizes after a few epochs, which means the model isn't only learning meaningful patterns, but it also quickly converges into a stable solution, which means, given the current configuration and dataset, it's likely close to its performance ceiling. And because it reaches it quickly, 50 epochs is most likely unnecessary.

<p align="center">
  <img src="./model accuracy loss history.png" alt="Model Accuracy and Loss History" width="90%" />
  <br>
  <em>Figure 4. Training and validation accuracy and loss across epochs.</em>
</p>

_It is important to note that the test accuracy reported by Keras uses the default threshold of 0.50, while the classification report was generated using the adjusted threshold of 0.35. The accuracy values are not directly identical._


The recall achieved for the **Dropout** class shows that the model successfully identifies approximately **77%** of the students who actually dropped out, equivalent to around 218 out of 284 dropout cases.

On the other hand, the model achieved an **AUC of 0.81**, indicating a decent ability to distinguish between the target classes across different classification thresholds.

<p align="center">
  <img src="./confusion matrix and roc.png" alt="Confusion Matrix and ROC" width="45%" />
  <br>
  <em>Figure 5. Confusion Matrix and ROC Curve of the Logistic Regression model.</em>
</p>

The model did not achieve the same Accuracy or AUC reported by Hashim et al. (2020). This difference is likely explained by differences in dataset structure, target definition, class distribution, feature selection, or hyperparameter configuration.

As the model already appears to reach a relatively stable level of performance, future work should focus on improving its ability to better distinguish between the **Graduate** and **Dropout** classes, rather than solely pursuing higher recall or overall performance results.

---

## Logistic Regression Model (Refinement Attempt 1: Random Oversampling)

[Google Colab Notebook](https://colab.research.google.com/drive/1nuvm3LFewGHWe5JLWyW3MC4tEdfFl3ik?usp=sharing)

Based on the findings reported by Wongvorachan, He, and Bulut (2023), an attempt was made to improve the model by reducing class imbalance through the use of **Random Oversampling (ROS)**.

The objective was to to increase the representation of the minority class (Dropout) during training.

The model did not only showed marginal improvements on **recall**, but it also diminished on **precision**, **F1-score**, and **stability**.

These results suggest that the current level of class imbalance is not significantly limiting the model's predictive performance.

---

## Logistic Regression Model (Refinement Attempt 2: SMOTE-NC)

[Google Colab Notebook](https://colab.research.google.com/drive/1qNMMk3W2h5FRHkToIwyXxmm9iVH-oIIz?usp=sharing)

Following the recommendations presented in the same study, a second refinement experiment was conducted using **SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous Features)**.

It generates synthetic samples for the minority class, instead of duplicating existing observations like ROS, with the goal of improving class representation while reducing the risk of overfitting associated with traditional oversampling methods.

However, the results obtained with SMOTE-NC were inferior to both the baseline model and the ROS experiment.

## Conclusions

Neither of the two experiments showed that the class imbalance is the primary factor for limiting the performance of the model. This likely comes due to having a moderate class distribution (approximately 60% Graduate and 40% Dropout), a less bias distribution than the papers.

| Model    | Precision (Dropout) | Recall (Dropout) | F1 (Dropout) |   AUC |
| -------- | ------------------: | ---------------: | -----------: | ----: |
| Baseline |                0.62 |             0.77 |         0.69 | 0.808 |
| ROS      |                0.57 |             0.82 |         0.68 | 0.815 |
| SMOTE-NC |                0.55 |             0.82 |         0.66 | 0.809 |

The training history shows that the training and validation metrics remain close throughout most epochs, with almost no underfitting or overfitting. The final model already achieves a recall of approximately 70% for Graduate students and 77% for Dropout students, maintaining the slight bias toward identifying students at risk of dropping out that is needed.

Through this and the marginal improvements observed in the previous attempts, we can infer that the Logistic Regression model is already close to its performance limit achievable with the current dataset and feature set.

To obtain better performance, it would most likely be necessary to change or expand the current dataset information, as the LR model has already been shown to be the best option among a variety of different algorithms according to the study conducted by Hashim et al. (2020).

Due to this, no further attempts to improve the model were made.

---

## Model Testing

The final model can be tested and used for predictions through the following Google Colab notebook:

🔗 https://colab.research.google.com/drive/1R0aSawSsxE3RkHDhGentxrZI2gTyhMsC?usp=sharing

---

## References

alejandraa-cruiz. (n.d.). MushroomClassification [Source code]. GitHub. Retrieved May 31, 2026, from https://github.com/alejandraa-cruiz/MushroomClassification

Hashim, A. S., Awadh, W. A., & Hamoud, A. K. (2020). Student performance prediction model based on supervised machine learning algorithms. IOP Conference Series: Materials Science and Engineering, 928(3), Article 032019. https://doi.org/10.1088/1757-899X/928/3/032019

Wongvorachan, T., He, S., & Bulut, O. (2023). A comparison of undersampling, oversampling, and SMOTE methods for dealing with imbalanced classification in educational data mining. Information, 14(1), 54. https://doi.org/10.3390/info14010054
