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

## Data Preprocessing

As the goal of the project is to create a model capable of making an early prediction of whether a student is at risk of dropping out, all attributes related to the faculty of origin, macroeconomic factors of the country of origin, and academic attributes not related to the students' academic starting point were removed.

Since the target variable includes **"Enrolled"**, a state that neither indicates whether students are at risk of dropping out nor whether they have a strong chance of graduating, it was decided to remove all rows containing this value.

Finally, the target variable was encoded into numerical labels with **LabelEncoder**, and **One-Hot Encoding** was used to transform the numeric labels into a numeric vector.

---

## Split

The data was split into **80% training** and **20% testing**, while maintaining similar proportions for the **target** variable.

---

## References

Wongvorachan, T., He, S., & Bulut, O. (2023). *A comparison of undersampling, oversampling, and SMOTE methods for dealing with imbalanced classification in educational data mining*. **Information, 14**(1), 54. https://doi.org/10.3390/info14010054
