# 🎓 Student Grade Predictor with Linear Regression
This project uses a linear regression model to predict students' final grades (G3) based on factors such as their previous grades (G1, G2), study time, number of past class failures, and absences. The model is trained and evaluated using scikit-learn, and the best-performing model is saved using Python's pickle module.

# 📁 Dataset
The project uses the UCI Student Performance Dataset, specifically student-mat.csv, which contains data related to student achievement in secondary education (Math course).

Each student is described by various features including:
G1, G2: First and second period grades
studytime: Weekly study time
failures: Number of past class failures
absences: Number of school absences
G3: Final grade (target)

# 🛠️ Features Used
Only the following columns are used for prediction:
G1
G2
studytime
failures
absences

Target variable: G3

# 🧠 How It Works
Load the dataset and select relevant columns.
Split the data into input (X) and target (Y).
Perform multiple train-test splits to find the best performing model.
Train a LinearRegression model from scikit-learn.
Save the model with the highest accuracy using pickle.
Load the model and use it to make predictions.
Display predictions alongside actual values and input features.
Plot a scatter plot of absences vs. final grade for visualization.

# 📦 Dependencies
Make sure you have the following Python packages installed:
pip install pandas numpy scikit-learn matplotlib

# 📊 Sample Output

acc = 0.86
acc = 0.89
acc = 0.88
...
15.6 [15 14 2 0 4] 15
13.1 [10 11 3 1 6] 14
...
A scatter plot of absences vs G3 will also be displayed.

# 📁 Files
student_grade_predictor.py: Main script to train and test the 0model
student-mat.csv: Dataset file (you must download this manually)
model.pickle: Serialized best-trained model (accuracy > 94%)

# 📈 Future Improvements
Test with additional features from the dataset for better accuracy
Add model evaluation metrics like MAE or RMSE
Create a simple web UI using Flask or Streamlit
# 📜 License
This project is licensed under the MIT License.
