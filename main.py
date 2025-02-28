from ucimlrepo import fetch_ucirepo

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# metadata
print(cdc_diabetes_health_indicators.metadata)

# variable information
print(cdc_diabetes_health_indicators.variables)

from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns

# sns.set_theme(style="ticks")
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

df1 = X #[['Age', 'BMI', 'MentHlth', 'PhysHlth']]
df2 = y["Diabetes_binary"]
df1["Diabetes_binary"] = df2
#sns.pairplot(df1, hue="Diabetes_binary", plot_kws={'alpha': 0.2})
#sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})

sns.histplot(data=df1, x="BMI", hue="Diabetes_binary", multiple="dodge", shrink=.8, bins=7)
sns.histplot(data=df1, x="MentHlth", hue="Diabetes_binary", multiple="dodge", shrink=.8, bins=7)
sns.histplot(data=df1, x="PhysHlth", hue="Diabetes_binary", multiple="dodge", shrink=.8, bins=7)
sns.histplot(data=df1, x="Age", hue="Diabetes_binary", multiple="dodge", shrink=.8, bins=13)

from sklearn.metrics import confusion_matrix as cm

features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

counts = df1['Diabetes_binary'].value_counts()
print(counts)

for feature in features:
  print(feature)
  result = cm(df1['Diabetes_binary'], df1[feature])
  print("For Diabetes = No")
  value = result[0][0]/(result[0][0]+result[0][1])
  print(value, 1.0 - value)
  print("For Diabetes = Yes")
  value = result[1][0]/(result[1][0]+result[0][1])
  print(value, 1.0 - value)
  value0 = result[0][0]+result[1][0]
  value1 = result[0][1]+result[1][1]
  print(value0/(value0+value1), value1/(value0+value1))
  print("\n\n")

from multiprocessing import Array
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Decision Tree",
    "Logistic Regression",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(n_neighbors=10), # 10 neighbors
    LinearSVC(random_state=0),
    DecisionTreeClassifier(max_depth=None, random_state=0),
    LogisticRegression(random_state=0),
    RandomForestClassifier(
        max_depth=None, n_estimators=100, random_state=0
    ),
    AdaBoostClassifier(algorithm="SAMME", random_state=0),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
array = []
cv_folds = 10

for i in range(0, len(classifiers)):
  clf = classifiers[i]
  scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='roc_auc')
  array.append(scores.mean())
  print("Ran ", names[i], " ", scores.mean())

from ucimlrepo import fetch_ucirepo
import pandas as pd

cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets
X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

import shap

clf = LinearSVC(random_state=0)

# Train the model
clf.fit(X_train, y_train)

explainer = shap.Explainer(clf.predict, X_test)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

import seaborn as sns
%matplotlib inline

# calculate the correlation matrix on the numeric columns
corr = X.select_dtypes('number').corr()

# plot the heatmap
sns.heatmap(corr)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = [
    KNeighborsClassifier(n_neighbors=10), # 10 neighbors
    LinearSVC(random_state=0),
    DecisionTreeClassifier(max_depth=None, random_state=0),
    LogisticRegression(random_state=0),
    RandomForestClassifier(
        max_depth=None, n_estimators=100, random_state=0
    ),
    AdaBoostClassifier(algorithm="SAMME", random_state=0),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
# Create a classifier and fit it to the training data
model = LogisticRegression()
model2 = KNeighborsClassifier()
model3 = LinearSVC()
model4 = DecisionTreeClassifier()
model5 = RandomForestClassifier()
model6 = AdaBoostClassifier()
model7 = GaussianNB()
model8 = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)
model7.fit(X_train, y_train)
model8.fit(X_train, y_train)

# Get predicted probabilities for the test data
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_proba2 = model2.predict_proba(X_test)[:, 1]
# y_pred_proba3 = model3.predict(X_test)
clf = CalibratedClassifierCV(model3)
clf.fit(X_train, y_train)
y_pred_proba3 = clf.predict_proba(X_test)[:, 1]
y_pred_proba4 = model4.predict_proba(X_test)[:, 1]
y_pred_proba5 = model5.predict_proba(X_test)[:, 1]
y_pred_proba6 = model6.predict_proba(X_test)[:, 1]
y_pred_proba7 = model7.predict_proba(X_test)[:, 1]
y_pred_proba8 = model8.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
print(y_pred_proba3)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba2)
roc_auc2 = auc(fpr2, tpr2)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred_proba3)
roc_auc3 = auc(fpr3, tpr3)
fpr4, tpr4, thresholds4 = roc_curve(y_test, y_pred_proba4)
roc_auc4 = auc(fpr4, tpr4)
fpr5, tpr5, thresholds5 = roc_curve(y_test, y_pred_proba5)
roc_auc5 = auc(fpr5, tpr5)
fpr5, tpr5, thresholds5 = roc_curve(y_test, y_pred_proba5)
roc_auc5 = auc(fpr5, tpr5)
fpr6, tpr6, thresholds6 = roc_curve(y_test, y_pred_proba6)
roc_auc6 = auc(fpr6, tpr6)
fpr7, tpr7, thresholds7 = roc_curve(y_test, y_pred_proba7)
roc_auc7 = auc(fpr7, tpr7)
fpr8, tpr8, thresholds8 = roc_curve(y_test, y_pred_proba8)
roc_auc8 = auc(fpr8, tpr8)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='Logistic Regression')
plt.plot(fpr2, tpr2, color='black', label='K Neighbors Classifier')
plt.plot(fpr3, tpr3, color='red', label='Linear SVC')
plt.plot(fpr4, tpr4, color='cyan', label='Decision Tree')
plt.plot(fpr5, tpr5, color='green', label='Random Forest')
plt.plot(fpr6, tpr6, color='purple', label='AdaBoost')
plt.plot(fpr7, tpr7, color='pink', label='Gaussian')
plt.plot(fpr8, tpr8, color='gray', label='QDA')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig('foo.png')

thresholds_data = pd.DataFrame(data={"TPR": tpr3, "FPR": fpr3, "Thresholds": thresholds3})
thresholds_data

import pickle
from google.colab import files

print(clf)

#save model
filename='saved_model.sav'
pickle.dump(clf,open(filename,'wb'))
#files.download(filename)

#load model and make predictions from loaded model
model = pickle.load(open(filename,'rb'))
final_probs=model.predict_proba(X_test)
print(final_probs)

from sklearn.metrics import confusion_matrix as cm

threshold = 0.12079556854343798
predicted_labels = np.array([1 if p >= threshold else 0 for p in final_probs[:, 1]])
cm(y_test, predicted_labels)
