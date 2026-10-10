# Outlet Sales Prediction Using Machine Learning 📊

## 📌 Project Overview
This project predicts retail outlet sales using machine learning regression algorithms. It applies data preprocessing, feature encoding, feature scaling, and model optimization to estimate `Item_Outlet_Sales` based on product and outlet characteristics.

## 🎯 Objectives
- Analyze retail product and outlet sales data.
- Handle missing values in the dataset.
- Encode categorical variables into numerical features.
- Apply feature scaling to improve KNN regression.
- Train and evaluate regression models.
- Optimize model parameters using GridSearchCV and RandomizedSearchCV.

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📂 Dataset
The notebook uses a CSV dataset named `KNN_reg_outlet_sales - KNN_reg_outlet_sales.csv`.

The dataset contains product and outlet-related information, including:
- `Item_Identifier`
- `Item_Weight`
- `Item_Fat_Content`
- `Item_Type`
- `Item_MRP`
- `Outlet_Identifier`
- `Outlet_Size`
- `Outlet_Location_Type`
- `Outlet_Type`
- `Item_Outlet_Sales`

**Target variable:** `Item_Outlet_Sales`

## ⚙️ Project Workflow
1. Import the required Python libraries.
2. Load and explore the dataset.
3. Check and handle missing values.
4. Encode categorical variables using numerical mapping and one-hot encoding.
5. Separate input features and the target variable.
6. Apply MinMaxScaler for feature scaling.
7. Split the dataset into training and testing sets.
8. Train a K-Nearest Neighbors (KNN) Regressor.
9. Evaluate the model using regression metrics.
10. Tune hyperparameters using GridSearchCV and RandomizedSearchCV.
11. Explore Decision Tree and Random Forest regression models.
12. Compare model performance using the R² score.

## 🤖 Machine Learning Algorithms

### 1. K-Nearest Neighbors (KNN) Regression
Predicts sales by considering the target values of nearby data points.

### 2. Decision Tree Regression
Predicts sales using decision rules learned from the dataset.

### 3. Random Forest Regression
Combines multiple decision trees to generate sales predictions.

## 📈 Model Evaluation
The notebook uses these evaluation metrics:

- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the target variable's units.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **R² Score:** Measures how well the model explains variation in sales.

Actual performance values can be added after running the notebook.

## 🚀 How to Run the Project

### 1. Clone your repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Outlet-Sales-Predictor
```

Replace the URL with your actual GitHub repository URL.

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Add the dataset
Place the required CSV file in the notebook's working directory.

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

Open `KNN Regression.ipynb` and run the cells in order.

## 📁 Project Structure
```text
Outlet-Sales-Predictor/
│
├── KNN Regression.ipynb
├── KNN_reg_outlet_sales - KNN_reg_outlet_sales.csv
└── README.md
```

## 🔮 Future Improvements
- Build an interactive sales prediction application using Streamlit.
- Compare regression models using consistent train-test splits.
- Perform feature importance analysis.
- Improve predictions through systematic hyperparameter tuning.
- Deploy the best-performing model for practical use.

## ⚠️ Disclaimer
This project is intended for educational purposes and machine learning practice.

## 👨‍💻 Author
**Sanket Bacche**
**LinkedIn link-** linkedin.com/in/sanket-bacche-00320a380

Artificial Intelligence and Data Science Student

## ⭐ Conclusion
This project demonstrates how regression algorithms can be applied to retail sales prediction, with a focus on data preprocessing, model evaluation, and hyperparameter optimization.
