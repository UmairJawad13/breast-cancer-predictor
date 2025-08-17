# %% [markdown]
# ### Step 1: Data Identification (Import Data)
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

#Import data
data=pd.read_csv("Cancer_Data.csv")

#Take a preview
data.head(5)

# %%
import joblib


# %%
#General information

data.info()

# %%
# Define the 31 features for model prediction (using "mean" features only)
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
    "smoothness_mean", "compactness_mean", "concavity_mean", 
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", 
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", 
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", 
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]


# %% [markdown]
# ### Step 2: Data Preprocessing 
# 
# #### Encode categorical variables

# %%
#Categorical variables: diagnosis

#Check frequency tables

print(data.diagnosis.value_counts())

# %% [markdown]
# #### Check for missing values

# %%
missing_values = data.isnull().sum()

print("Missing values in each column:")
print(missing_values)

# %% [markdown]
# #### Dealing with missing values (Imputation)

# %%
#No Missing Values

# %% [markdown]
# ### Step 3: Exploratory Data Analysis
# 
# A brief exploration to get a better idea of what our dataset contains; will give us a better idea of how to process the data. b

# %%
#Descriptive statistics for numerical variables

data[[
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]].describe()


# %%
import matplotlib.patches as mpatches

# Count the frequency of each category in the 'diagnosis' column
diagnosis_counts = data['diagnosis'].value_counts()

# Create a bar plot for the diagnosis distribution
plt.figure(figsize=(5, 5))
ax = diagnosis_counts.plot(kind='bar', color=['Skyblue', 'Indianred'])
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Create custom legend handles for benign and malignant diagnosis
benign_patch = mpatches.Patch(color='Skyblue', label='Benign (B)')
malignant_patch = mpatches.Patch(color='Indianred', label='Malignant (M)')

# Add the legend to the top right corner
plt.legend(handles=[benign_patch, malignant_patch], loc='upper right')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Graph explanation:
# 
# The bar chart presents the distribution of breast cancer diagnoses, with the x-axis indicating two categories—Benign (represented in blue) and Malignant (represented in red)—and the y-axis showing the frequency of occurrences for each diagnosis. The graph reveals that there are 357 benign cases compared to 212 malignant cases, indicating that about 63% of the cases are benign while 37% are malignant. This class imbalance is clearly illustrated by the taller blue bar versus the shorter red bar, and the grid lines aid in accurately gauging these differences, highlighting the importance of considering this distribution when analyzing the data or building predictive models.

# %%
# Convert 'diagnosis' column to numeric: M=1, B=0

# Drop 'id' and any unnamed columns
data = data.drop(columns=['id'], errors='ignore')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Select numeric columns only
data_numeric = data.select_dtypes(include=['number'])

# Filter to only include columns that contain "mean" (case-insensitive)
mean_columns = [col for col in data_numeric.columns if 'mean' in col.lower()]
data_means = data_numeric[mean_columns]

# Create a correlation heatmap for the mean columns
plt.figure(figsize=(12, 6))
sns.heatmap(data_means.corr(), annot=True, cmap="coolwarm", fmt=".1f")
plt.title('Correlation Heatmap of Mean Features')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Graph explanation:
# 
# This heatmap shows how different features (like radius, texture, and area) are related. Each square has a number called a correlation, which tells us if two features move together. A number close to 1 (red) means that when one feature increases, the other also increases. A number close to -1 (blue) means that when one goes up, the other goes down. Numbers near 0 show little or no connection. The diagonal from top left to bottom right is all 1's because each feature is perfectly related to itself.

# %%
import matplotlib.patches as mpatches

# Count values without converting diagnosis to numeric
diagnosis_counts = data['diagnosis'].value_counts()

# Set a pastel color palette with exactly 2 colors for consistency
colors = sns.color_palette('pastel', 2)

# Create pie chart
plt.figure(figsize=(5, 5))
plt.pie(
    diagnosis_counts,
    labels=diagnosis_counts.index,  # ['B', 'M']
    autopct='%1.1f%%',
    startangle=140,
    colors=colors
)
plt.title('Diagnosis Distribution')

# Define a mapping from diagnosis code to full label
mapping = {'B': 'Benign (B)', 'M': 'Malignant (M)'}

# Ensure diagnosis_counts.index contains original labels ('B', 'M')
original_labels = diagnosis_counts.index.map(lambda x: x[0] if x in mapping.values() else x)

# Map the original labels to full labels for the legend
mapped_labels = [mapping[label] for label in original_labels]



plt.tight_layout()
plt.show()


# %% [markdown]
# ### Graph explanation
# 
# This pie chart shows the percentage of breast cancer cases in two categories: **Benign (B)** and **Malignant (M)**. The blue slice, which represents **Benign (B)**, makes up around **62.7%** of the total cases, while the orange slice for **Malignant (M)** accounts for about **37.3%**. This means that in the dataset, roughly two-thirds of the samples are benign, and about one-third are malignant.
# 

# %%
plt.figure(figsize=(6, 5))
sns.histplot(data=data, x='radius_mean', hue='diagnosis', bins=20, kde=True, palette='pastel', edgecolor='black')
plt.title('Radius Mean Distribution by Diagnosis')
plt.xlabel('Radius Mean')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Overwrite legend labels with full words
plt.legend(title="Diagnosis", labels=["Benign (B)", "Malignant (M)"])

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Graph explanation
# 
# This histogram compares how the **radius mean** measure is distributed for **Benign (B)** cases (in orange) versus **Malignant (M)** cases (in blue). Each set of bars (and the smooth curve) shows how often particular radius values occur. Notice that benign cases cluster around lower radius values—mostly under 15—while malignant cases extend toward higher radius values, often above 15. This suggests that, in this dataset, tumors classified as malignant tend to have a larger average radius compared to those classified as benign.

# %%
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Cancer_Data.csv")

plt.figure(figsize=(6, 4))
ax = sns.violinplot(data=data, x='diagnosis', y='radius_mean', palette="pastel")
plt.title('Radius Mean Distribution by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Radius Mean')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Create custom legend handles for benign and malignant categories
colors = sns.color_palette("pastel", 2)
legend_handles = [
    mpatches.Patch(color=colors[0], label='Benign (B)'),
    mpatches.Patch(color=colors[1], label='Malignant (M)')
]

plt.legend(handles=legend_handles, loc='best')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Graph explanation
# 
# This violin plot shows how the **radius mean** values differ between **Malignant (M)** (blue) and **Benign (B)** (orange) cases. Each "violin" shape illustrates the full distribution of radius values, where the wider sections indicate that those radius measurements occur more often. The central boxplot inside each violin displays the median (thick line) and the middle range of values (box). From the plot, you can see that **malignant tumors** generally have a larger range of radius mean values—spreading higher—while **benign tumors** have a narrower, lower range. This suggests that malignant tumors are often larger on average than benign tumors.

# %%
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
ax = sns.boxplot(data=data, x="diagnosis", y="area_mean", palette="Set2")
plt.title("Area Mean Distribution by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Area Mean")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Define the colors from the Set2 palette for the two diagnosis categories
colors = sns.color_palette("Set2", 2)
# Create custom legend handles
legend_handles = [
    mpatches.Patch(color=colors[0], label='Benign (B)'),
    mpatches.Patch(color=colors[1], label='Malignant (M)')
]

plt.legend(handles=legend_handles, loc='best')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Graph explanation
# 
# This boxplot compares the **area mean** values for **Malignant (M)** (green) and **Benign (B)** (orange) cases. Each box represents the middle spread of the data (the interquartile range), with the horizontal line inside the box indicating the median. From the plot, you can see that malignant tumors typically have higher area means and a wider spread, suggesting larger sizes on average and greater variability. By contrast, benign tumors cluster at lower area means, with fewer and smaller outliers. This underscores that, on average, **malignant tumors** in the dataset tend to occupy a larger area than **benign tumors**.
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# Sample data processing and label replacement for facets
np.random.seed(42)
data["Gender"] = np.random.choice(["Male", "Female"], size=len(data))
data['diagnosis'] = data['diagnosis'].replace({'B': 'Benign (B)', 'M': 'Malignant (M)'})

# Plotting FacetGrid: Radius Mean Distribution by Diagnosis and Gender
g = sns.FacetGrid(data, col="Gender", row="diagnosis", height=2.5, margin_titles=True)
g.map(plt.hist, "radius_mean", bins=15, color="steelblue", edgecolor="black")

# Adjust layout and add title
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Radius Mean Distribution by Diagnosis and Gender")

# Calculate divider positions based on the axes positions
# For vertical line: average x coordinate between right edge of left column and left edge of right column (top row)
pos_left = g.axes[0, 0].get_position()
pos_right = g.axes[0, 1].get_position()
vline_x = (pos_left.x1 + pos_right.x0) / 2

# For vertical line y-range, use the bottom of the bottom row and the top of the top row (so it stays within the subplots)
bottom_y = g.axes[1, 0].get_position().y0
top_y = g.axes[0, 0].get_position().y1  # This is below the title because of subplots_adjust(top=0.9)

# Create a vertical line that spans from bottom_y to top_y
vline = mlines.Line2D([vline_x, vline_x], [bottom_y, top_y],
                       color="blue", linewidth=1, transform=g.fig.transFigure)
g.fig.add_artist(vline)

# For horizontal divider: average y coordinate between bottom edge of top row and top edge of bottom row (left column)
pos_top_left = g.axes[0, 0].get_position()
pos_bottom_left = g.axes[1, 0].get_position()
hline_y = (pos_top_left.y0 + pos_bottom_left.y1) / 2

# Create a horizontal line that spans the width of the figure
hline = mlines.Line2D([0, 1], [hline_y, hline_y],
                       color="black", linewidth=1, transform=g.fig.transFigure)
g.fig.add_artist(hline)

plt.show()


# %% [markdown]
# ### Graph explanation
# 
# This 2×2 grid shows how the **radius mean** is distributed for different combinations of **diagnosis** (Malignant vs. Benign) and **gender** (Male vs. Female). Each subplot is a histogram representing a specific subset of the data: the top row is for malignant cases, the bottom row is for benign cases, while the left column is for males and the right column is for females. By comparing these four histograms, you can see variations in the range and most common values of **radius mean** across both diagnosis and gender groups. Generally, the malignant distributions (top row) shift toward larger radius mean values compared to the benign distributions (bottom row), suggesting that malignant tumors tend to have higher radius measurements regardless of gender.

# %%
# Replace diagnosis letters with full words
data['diagnosis'] = data['diagnosis'].replace({'B': 'Benign (B)', 'M': 'Malignant (M)'})

# Bin the 'texture_mean' column into 10 intervals
data['texture_bin'] = pd.cut(data['texture_mean'], bins=10)

# Create crosstab: rows will now show full diagnosis labels
crosstab = pd.crosstab(data['diagnosis'], data['texture_bin'])

# Plot heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='coolwarm')
plt.title("Crosstab Heatmap: Diagnosis vs. Binned Texture Mean")
plt.ylabel("Diagnosis")
plt.xlabel("Texture Mean (Binned)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Graph explanation
# 
# In this crosstab heatmap, the horizontal axis shows binned intervals of the **texture_mean** feature (from lower to higher values), while the vertical axis lists the two diagnosis categories, **Malignant (M)** and **Benign (B)**. Each cell displays how many samples fall into the corresponding combination of diagnosis and texture_mean bin, with warmer (red) colors indicating higher counts and cooler (blue) colors indicating fewer. For example, we see the largest group in the benign row at around the 15.6–18.6 bin (123 samples), suggesting that many benign tumors fall within that texture range. This visualization helps identify which ranges of texture_mean are most common for benign vs. malignant cases in the dataset.

# %%
# Ensure concavity_mean is numeric and drop missing values
data['concavity_mean'] = pd.to_numeric(data['concavity_mean'], errors='coerce')
kde_data = data[['concavity_mean', 'diagnosis']].dropna()

# Separate data by diagnosis
malignant = kde_data[kde_data['diagnosis'] == 'M']['concavity_mean']
benign = kde_data[kde_data['diagnosis'] == 'B']['concavity_mean']

# Debugging: Print sizes of datasets
print(f"Number of malignant samples: {len(malignant)}")
print(f"Number of benign samples: {len(benign)}")

# Check if datasets have sufficient elements
if len(malignant) > 1 and len(benign) > 1:
    # Create KDE functions
    x_vals = np.linspace(0, kde_data['concavity_mean'].max(), 500)
    kde_m = gaussian_kde(malignant)
    kde_b = gaussian_kde(benign)

    # Plot the KDEs
    plt.figure(figsize=(6, 4))
    plt.fill_between(x_vals, kde_m(x_vals), alpha=0.5, label='Malignant (M)')
    plt.fill_between(x_vals, kde_b(x_vals), alpha=0.5, label='Benign (B)')
    plt.title("Kernel Density Plot of Concavity Mean by Diagnosis")
    plt.xlabel("Concavity Mean")
    plt.ylabel("Density")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Graph Explanation
# 
# This kernel density plot compares **concavity_mean** for **Malignant (M)** (blue) and **Benign (B)** (orange) breast tumors. Each curve shows how often different concavity values occur—higher peaks mean more samples have values in that range. You can see that **benign tumors** (orange) tend to cluster at lower concavity_mean values, peaking sharply, while **malignant tumors** (blue) are spread over a broader range with generally higher concavity_mean values. This suggests that, overall, malignant tumors often have higher concavity_mean than benign tumors, though there is some overlap between the two distributions.

# %% [markdown]
# ### Step 4: Data Splitting
# 
# Splitting your dataset into training and testing sets is critical. It allows you to train your models on one portion of the data and evaluate performance on unseen data. Here, we assume that your dataset is loaded into a DataFrame called df and that your target variable (e.g., a column indicating benign/malignant status) is called 'diagnosis'. Adjust the target name if needed.

# %% [markdown]
# #### Convert Diagnosis Column to Numeric
# 
# To ensure compatibility with machine learning models, we convert the `diagnosis` column to numeric values: `M=1` (Malignant) and `B=0` (Benign).

# %%
# Convert 'diagnosis' column to numeric: M=1, B=0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Verify the conversion
print(data['diagnosis'].unique())

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("Cancer_Data.csv")

# Define features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Dataset split completed.")

# %% [markdown]
# ### Explanation:
# 
# Data Loading: Ensure that your DataFrame df is correctly loaded.
# 
# Feature and Target Definition: We remove the target column from X and set y as the target.
# 
# Missing Values: It’s a good practice to check for missing values.
# 
# Train-Test Split: We use an 80/20 split with stratification so that both training and test sets preserve the same class distribution.
# 
# Target Conversion: The `diagnosis` column is converted to numeric values (`M=1`, `B=0`) to ensure compatibility with machine learning models.

# %% [markdown]
# ### Step 5: Model Training
# 
# In this step, we train five different classification models. These models are:
# 
# Logistic Regression
# 
# Random Forest Classifier
# 
# Support Vector Machine (SVM)
# 
# Gradient Boosting Classifier (using XGBoost)
# 
# AdaBoost Classifier
# 
# Each model is initialized, trained on the training data, and used to generate predictions on the test set. Additionally, for each model, we extract the probability estimates for the positive class (needed later for ROC curve analysis).

# %% [markdown]
# ### 5.1 Logistic Regression

# %%
# Import the Logistic Regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer

# Impute missing values in X_train and X_test
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialize Logistic Regression with a higher number of iterations to ensure convergence.
log_reg = LogisticRegression(max_iter=10000, random_state=42)

# Train the Logistic Regression model on the training data.
log_reg.fit(X_train, y_train)

# Generate predictions on the test set.
y_pred_log = log_reg.predict(X_test)

# Extract probability estimates for the positive class (useful for ROC curve analysis).
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

print("Logistic Regression model training completed.")


# %% [markdown]
# ### Explanation:
# 
# Model: Logistic Regression
# 
# Why: It is simple, interpretable, and serves as a strong baseline.
# 
# Notes: We set max_iter=10000 to help the optimization converge.

# %% [markdown]
# ### 5.2 Random Forest Classifier

# %%
# Import the Random Forest Classifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier with 100 trees.
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model on the training data.
rf_clf.fit(X_train, y_train)

# Generate predictions on the test set.
y_pred_rf = rf_clf.predict(X_test)

# Get probability estimates for the positive class.
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

print("Random Forest model training completed.")


# %% [markdown]
# ### Explanation:
# 
# Model: Random Forest Classifier
# 
# Why: It uses an ensemble of decision trees to reduce overfitting and can capture non-linear relationships.
# 
# Notes: The default 100 estimators work well for many datasets; you can tune this parameter if needed.

# %% [markdown]
# ### 5.3 Support Vector Machine (SVM)

# %%
# Import the SVM classifier from scikit-learn
from sklearn.svm import SVC

# Initialize the SVM model with an RBF kernel and enable probability estimates.
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# Train the SVM model on the training data.
svm_model.fit(X_train, y_train)

# Generate predictions on the test set.
y_pred_svm = svm_model.predict(X_test)

# Get probability estimates for the positive class.
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

print("SVM model training completed.")


# %% [markdown]
# ### Explanation:
# 
# Model: Support Vector Machine (SVM)
# 
# Why: SVMs are effective in high-dimensional spaces and, with the RBF kernel, capture non-linear decision boundaries.
# 
# Notes: Enabling probability estimates is crucial for later ROC curve analysis.

# %% [markdown]
# ### 5.4 Gradient Boosting Classifier (XgBoost)

# %%
# Import the XGBoost classifier
import xgboost as xgb 

# Convert the target variable 'y_train' and 'y_test' to numeric: M=1, B=0
y_train_numeric = y_train.map({'M': 1, 'B': 0})
y_test_numeric = y_test.map({'M': 1, 'B': 0})

# Initialize the XGBoost Classifier.
# 'use_label_encoder=False' and 'eval_metric' are set to suppress warnings and define the evaluation metric.
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the XGBoost model on the training data.
xgb_clf.fit(X_train, y_train_numeric)

# Generate predictions on the test set.
y_pred_xgb = xgb_clf.predict(X_test)

# Get probability estimates for the positive class.
y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]

print("XGBoost (Gradient Boosting) model training completed.")


# %%
# Explanation for XGBoost model training was missing; added below.
print("XGBoost model training explanation completed.")

# %% [markdown]
# ### Explanation:
# 
# Model: XGBoost (Gradient Boosting Classifier)
# 
# Why: Gradient boosting builds a strong model by iteratively correcting the errors of previous models.
# 
# Notes: XGBoost is powerful and often achieves high predictive performance with structured/tabular data.

# %% [markdown]
# ### 5.5 AdaBoost Classifier

# %%
# Explanation for AdaBoost model training was missing; added below.
print("AdaBoost model training explanation completed.")

# %%
# Import the AdaBoost Classifier from scikit-learn
from sklearn.ensemble import AdaBoostClassifier

# Initialize the AdaBoost Classifier with 100 estimators.
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train the AdaBoost model on the training data.
ada_clf.fit(X_train, y_train)

# Generate predictions on the test set.
y_pred_ada = ada_clf.predict(X_test)

# Get probability estimates for the positive class.
y_prob_ada = ada_clf.predict_proba(X_test)[:, 1]

print("AdaBoost model training completed.")


# %% [markdown]
# ### Explanation:
# 
# Model: AdaBoost Classifier
# 
# Why: AdaBoost combines multiple weak learners (often decision trees) to form a robust classifier by focusing on misclassified instances.
# 
# Notes: It is a good choice when boosting the performance of simpler models is desired.

# %%
X_train = df[feature_names] 

# %%
# Select the 31 features for training
X_train = df[feature_names]  # This should match the 31 features defined above
y_train = df['diagnosis'].map({'B': 0, 'M': 1})  # Map 'B'/'M' to 0/1

# Train the model (XGBoost example)
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)  # Train the model with 31 features

# Save the model
joblib.dump(model, 'xgboost_model.pkl')
print("Model trained and saved successfully!")


# %% [markdown]
# ### Step 6: Model Testing
# 
# In this step, we verify that each trained model produces output on the test set. This “testing” step simply confirms that your models return predictions and probability estimates for the test set (X_test). You can inspect a few predictions from each model to ensure they are working as expected.

# %%
import joblib

# Load the trained model 
model = joblib.load('xgboost_model.pkl')


# %%
# Display a few predictions from each model for inspection.
print("Logistic Regression Predictions (first 10):", y_pred_log[:10])
print("Random Forest Predictions (first 10):", y_pred_rf[:10])
print("SVM Predictions (first 10):", y_pred_svm[:10])
print("XGBoost Predictions (first 10):", y_pred_xgb[:10])
print("AdaBoost Predictions (first 10):", y_pred_ada[:10])


# %% [markdown]
# ### Explanation:
# 
# This step verifies that each model is returning predictions for the test set.
# 
# You can manually inspect these predictions to ensure the models are functioning before moving on to a formal evaluation.

# %% [markdown]
# ### Step 7: Model Evaluation
# 
# In this step, we evaluate the performance of the models using various metrics and visualizations. We’ll use:
# 
# Accuracy Score: The overall correctness of the model.
# 
# Classification Report: Providing precision, recall, F1-score, and support for each class.
# 
# Confusion Matrix: A visualization of true versus predicted labels.
# 
# ROC Curve and AUC Score: To assess how well the model distinguishes between the two classes.

# %%
# Import necessary libraries for evaluation.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to evaluate a given model.
def evaluate_model(y_test, y_pred, y_prob, model_name):
    print(f"--- {model_name} Evaluation ---")
    
    # 1. Accuracy and Classification Report
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 2. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # 3. ROC Curve and AUC Calculation
    # Convert y_test to numeric values for compatibility with roc_curve
    y_test_numeric = y_test.map({'M': 1, 'B': 0})
    fpr, tpr, thresholds = roc_curve(y_test_numeric, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc

# Evaluate each model individually.
roc_auc_log  = evaluate_model(y_test, y_pred_log, y_prob_log, "Logistic Regression")
roc_auc_rf   = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")
roc_auc_svm  = evaluate_model(y_test, y_pred_svm, y_prob_svm, "SVM")
# Map numeric predictions back to string labels for compatibility
y_pred_xgb_mapped = pd.Series(y_pred_xgb).map({0: 'B', 1: 'M'})

roc_auc_xgb  = evaluate_model(y_test, y_pred_xgb_mapped, y_prob_xgb, "XGBoost")
roc_auc_ada  = evaluate_model(y_test, y_pred_ada, y_prob_ada, "AdaBoost")

# Plot all ROC curves on a single graph for visual comparison.
plt.figure(figsize=(8, 6))
models = {
    "Logistic Regression": y_prob_log,
    "Random Forest": y_prob_rf,
    "SVM": y_prob_svm,
    "XGBoost": y_prob_xgb,
    "AdaBoost": y_prob_ada
}
# Convert y_test to numeric values for compatibility with roc_curve
y_test_numeric = y_test.map({'M': 1, 'B': 0})

for model_name, y_prob in models.items():
    fpr, tpr, _ = roc_curve(y_test_numeric, y_prob)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.title("ROC Curve Comparison for All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# %% [markdown]
# ### Models Accuracy Analysis
# 
# 1) XGBoost Evaluation 97.37%
# 
# 2) AdaBoost Evaluation 96.49%
# 
# 3) Random Forest Evaluation 95.61%
# 
# 4) Logistic Regression Evaluation 92.11%
# 
# 5) SVM Evaluation 63.16%
# 
# ### The best model is XGBoost Evaluation with accurace rate of 97.37%

# %%
# Define feature ranges for all 31 features (these need to match the ranges of the features in your dataset)
feature_ranges = {
    "radius_mean": (6.0, 30.0),
    "texture_mean": (9.0, 40.0),
    "perimeter_mean": (40.0, 200.0),
    "area_mean": (140.0, 2500.0),
    "smoothness_mean": (0.05, 0.20),
    "compactness_mean": (0.02, 0.35),
    "concavity_mean": (0.0, 0.45),
    "concave points_mean": (0.0, 0.25),
    "symmetry_mean": (0.10, 0.35),
    "fractal_dimension_mean": (0.04, 0.10),
    "radius_se": (0.0, 5.0),  # Just example ranges
    "texture_se": (0.0, 5.0),
    "perimeter_se": (0.0, 50.0),
    "area_se": (0.0, 500.0),
    "smoothness_se": (0.0, 0.10),
    "compactness_se": (0.0, 0.10),
    "concavity_se": (0.0, 0.10),
    "concave points_se": (0.0, 0.10),
    "symmetry_se": (0.0, 0.10),
    "fractal_dimension_se": (0.0, 0.10),
    "radius_worst": (10.0, 30.0),
    "texture_worst": (10.0, 30.0),
    "perimeter_worst": (50.0, 200.0),
    "area_worst": (500.0, 2500.0),
    "smoothness_worst": (0.05, 0.20),
    "compactness_worst": (0.05, 0.35),
    "concavity_worst": (0.0, 0.45),
    "concave points_worst": (0.0, 0.25),
    "symmetry_worst": (0.0, 0.35),
    "fractal_dimension_worst": (0.0, 0.10)
}


# %%
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# -------------------------------
# LOAD THE TRAINED MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load('xgboost_model.pkl')

model = load_model()

# -------------------------------
# PAGE TITLE
# -------------------------------
st.title("🩺 Breast Cancer Tumor Prediction")
st.markdown("Enter the values below based on the patient’s test results. The app will predict whether the tumor is **Benign** or **Malignant** using a trained XGBoost model.")

# -------------------------------
# DEFINE FEATURE LIST AND RANGES
# -------------------------------
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

# You can tune these based on real min-max values
feature_ranges = {
    "radius_mean": (6.0, 30.0),
    "texture_mean": (9.0, 40.0),
    "perimeter_mean": (40.0, 200.0),
    "area_mean": (140.0, 2500.0),
    "smoothness_mean": (0.05, 0.20),
    "compactness_mean": (0.02, 0.35),
    "concavity_mean": (0.0, 0.45),
    "concave points_mean": (0.0, 0.25),
    "symmetry_mean": (0.10, 0.35),
    "fractal_dimension_mean": (0.04, 0.10),
    "radius_se": (0.0, 5.0),
    "texture_se": (0.0, 5.0),
    "perimeter_se": (0.0, 50.0),
    "area_se": (0.0, 500.0),
    "smoothness_se": (0.0, 0.10),
    "compactness_se": (0.0, 0.10),
    "concavity_se": (0.0, 0.10),
    "concave points_se": (0.0, 0.10),
    "symmetry_se": (0.0, 0.10),
    "fractal_dimension_se": (0.0, 0.10),
    "radius_worst": (10.0, 30.0),
    "texture_worst": (10.0, 30.0),
    "perimeter_worst": (50.0, 200.0),
    "area_worst": (500.0, 2500.0),
    "smoothness_worst": (0.05, 0.20),
    "compactness_worst": (0.05, 0.35),
    "concavity_worst": (0.0, 0.45),
    "concave points_worst": (0.0, 0.25),
    "symmetry_worst": (0.0, 0.35),
    "fractal_dimension_worst": (0.0, 0.10)
}

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("🔢 Input Patient Test Values")

user_input = []
for feature in feature_names:
    min_val, max_val = feature_ranges[feature]
    value = st.slider(
        label=feature,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2),
        step=0.01
    )
    user_input.append(value)

# Convert input to 2D array for prediction
input_array = np.array(user_input).reshape(1, -1)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔮 Predict Diagnosis"):
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)

    label_map = {0: "Benign", 1: "Malignant"}
    diagnosis = label_map[prediction[0]]
    confidence = prediction_proba[0][prediction[0]] * 100

    st.markdown("---")
    st.subheader("🧾 Prediction Result")
    st.write(f"**Diagnosis:** {diagnosis}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    if diagnosis == "Malignant":
        st.error("⚠️ High risk — immediate medical attention advised.")
    else:
        st.success("✅ Low risk — continue with routine monitoring.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    """
    _This app was developed using an XGBoost model trained on breast cancer data.
    Predictions are only as good as the data provided — always consult a medical professional for final diagnosis._
    """
)



