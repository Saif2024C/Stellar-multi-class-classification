# Stellar-multi-class-classification

Project Overview
This project focuses on building machine learning models to classify astronomical objects into three categories:

Galaxy
Star
QSO (Quasi-Stellar Objects)

The objective is to predict the object class using machine learning techniques like:

SVM (Support Vector Machines)
XGBoost
K-Nearest Neighbors (KNN)
Dataset
The dataset includes the following features:

u: Ultraviolet magnitude
g, r, i, z: Optical magnitudes in different bands
class: Target column with categories ('GALAXY', 'STAR', 'QSO')
Data Source: stellar-classification-dataset-sdss17.zip

Requirements
To run this project, you need the following Python libraries:

pandas: Data loading and manipulation
numpy: Numerical operations
scikit-learn: Machine learning algorithms and evaluation
xgboost: Gradient boosting algorithm
matplotlib: Data visualization
seaborn: Statistical plotting
Install dependencies using:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
Project Structure
bash
Copy code
ML_astro/
├── data/
│   └── star_classification.csv   # Raw dataset file
├── main.py                       # Code for model training and evaluation
├── README.md                     # Project documentation
└── requirements.txt              # List of project dependencies
How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/ML_astro.git
cd ML_astro
Place the dataset star_classification.csv in the data/ folder.

Run the main script:

bash
Copy code
python main.py
Code Workflow
The project follows these key steps:

Load the Data:

The CSV file is loaded into a pandas DataFrame.
Target labels ('GALAXY', 'STAR', 'QSO') are converted into numeric values (0, 1, 2).
Data Preprocessing:

Features are standardized using StandardScaler.
Model Training:

SVM, XGBoost, and KNN classifiers are trained on the dataset.
Data is split into training and test sets using an 80-20 split.
Model Evaluation:

Accuracy, classification report, and confusion matrix are used for evaluation.
Sample Code
Here’s an example snippet for model training:

python
Copy code
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', probability=True) \n
svm_model.fit(X_train_scaled, y_train) 

# Evaluate
y_pred = svm_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
Results
SVM: Linear kernel used for robust separation.
XGBoost: Excellent performance for multi-class classification.
KNN: Baseline algorithm to compare results.
The final model accuracies and confusion matrices are printed in the terminal for evaluation.

Future Improvements
Hyperparameter tuning using GridSearchCV.
Incorporate more advanced models like Random Forest or Neural Networks.
Visualize decision boundaries for better understanding.

# Author
Developed as part of an academic machine learning project. Contributions and suggestions are welcome.
