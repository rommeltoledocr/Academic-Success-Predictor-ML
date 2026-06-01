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

Finally, the target variable was encoded into numerical labels. Since only two target categories remained after preprocessing, it was not necessary to use One-Hot Encoding. Instead, the labels were manually encoded.

---

## Train-Test Split

The dataset was divided into **80% training data** and **20% testing data**. The `stratify` parameter was used during the split to preserve the original class distribution of the target variable in both subsets, ensuring that the proportion of each class remained consistent across the training and testing sets.

---

## References

alejandraa-cruiz. (n.d.). MushroomClassification [Source code]. GitHub. Retrieved May 31, 2026, from https://github.com/alejandraa-cruiz/MushroomClassification

Hashim, A. S., Awadh, W. A., & Hamoud, A. K. (2020). Student performance prediction model based on supervised machine learning algorithms. IOP Conference Series: Materials Science and Engineering, 928(3), Article 032019. https://doi.org/10.1088/1757-899X/928/3/032019

Wongvorachan, T., He, S., & Bulut, O. (2023). A comparison of undersampling, oversampling, and SMOTE methods for dealing with imbalanced classification in educational data mining. Information, 14(1), 54. https://doi.org/10.3390/info14010054
