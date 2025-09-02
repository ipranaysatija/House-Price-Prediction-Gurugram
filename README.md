# House-Price-Prediction-Gurugram
This project focuses on predicting residential property prices in Gurugram using machine learning. It involves data cleaning, exploratory data analysis (EDA), feature engineering, and model building to understand how factors like location, size, number of bedrooms, and amenities impact housing prices. The goal is to provide an accurate and data-driven approach to real estate price estimation in one of India’s fastest-growing cities.
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/0*S0RD9WszR44AbOds.png" alt="OPS234 Radar" height="500">
</p>

## 📊 Dataset  
- The dataset contains details about houses in Gurugram.  
- Features include:  
  - Location 
  - Size (sq. ft.)  
  - Builder Details
  - BHK_Count
  - locality(Since it impacts the )
  - Price (target variable) and more.

Data Source: *https://www.kaggle.com/datasets/nikhilmehrahr26/gurgaon-real-estate-dataset?resource=download* 

## 🔑 Steps Involved  
1. **Data Preprocessing** – Handling missing values, removing outliers, and encoding categorical features.
    - Removing Outliers   
    ```python
    def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df
    ````
    - Scaling
    - Categorical Encoding
2. **Exploratory Data Analysis (EDA)** – Understanding price trends, correlations, and location-based patterns.  
<p align="center">
  <img src=https://github.com/ipranaysatija/House-Price-Prediction-Gurugram/blob/main/images/ggn%20hists.png?raw=true" alt="OPS234 Radar" height="500">
</p>
<p align="center">
  Exploratory Data Analysis (EDA)
</p>


3. **Visualization** - Understanding the various parameters and analyzing the sactter matrix to get a better understanding of feature importance.
<p align="center">
  <img src="https://github.com/ipranaysatija/House-Price-Prediction-Gurugram/blob/main/images/ggn%20scatter%20matrix.png?raw=true" alt="OPS234 Radar" height="500">
</p>
<p align="center">
  Gurgaon Data Scatter-Matrix
</p>

3. **Feature Engineering** – Creating meaningful features to improve model performance.
<p align="center">
  <img src="https://github.com/ipranaysatija/House-Price-Prediction-Gurugram/blob/main/images/Area%20vs%20Price%20Graph.png?raw=true" alt="Area vs Price Graph.png" height="500">
</p>
<p align="center">
  How area describes the final price
</p>  
Feature Tally:

``` python
feature_tally=grid_src.best_estimator_.feature_importances_
feature_tally
sorted(feature_tally,reverse=True)
```
Output:

```bash
 np.float64(0.9360037135563237),
 np.float64(0.046846081379410424),
 np.float64(0.00536405841068903),
 np.float64(0.0049369149111805535),
 np.float64(0.0019481502601063645),
 np.float64(0.0019455711521312654),
 np.float64(0.000681353228724848),
 np.float64(0.0005847238017292004),
 np.float64(0.000578381929947087),
 np.float64(0.0005557185965933308),
 np.float64(0.0003306009416534919),
 np.float64(0.00020956609762956007),
 np.float64(1.5165733881264313e-05)]
```
Depicts that the first few features are the major decising factors.


4. **Model Building** – Training regression models such as Linear Regression, Decision Tree, Random Forest, etc.  
```python
linReg=LinearRegression()
linReg.fit(trainer_prepared,train_labels)

TreeReg=DecisionTreeRegressor()
TreeReg.fit(trainer_prepared,train_labels)

RanReg=RandomForestRegressor()
RanReg.fit(trainer_prepared,train_labels)
```
5. **Model Evaluation** – Evaluating using metrics like MAE, and RMSE.
```python
lin_rmse = root_mean_squared_error(train_labels, lin_preds)
tree_rmse = root_mean_squared_error(train_labels, tree_preds)
forest_rmse = root_mean_squared_error(train_labels, forest_preds)
print("Linear Regression RMSE:", lin_rmse)
print("Decision Tree RMSE:", tree_rmse)
print("Random Forest RMSE:", forest_rmse)
```


## 💡 Applications  

- **Real Estate Agencies** – Estimate fair property values for clients.  
- **Home Buyers & Sellers** – Understand market-driven pricing before making decisions.  
- **Banks & Financial Institutions** – Assess property value for loan approvals and risk management.  
- **Urban Planners** – Analyze housing trends to support smart city development.  
- **Investors** – Identify undervalued or overvalued properties for investment decisions. 

## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/gurugram-house-price-prediction.git
   cd gurugram-house-price-prediction
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run The code for inference/training
    ```bash
    python3 script.py
    ```
4. Open the notebook:
```bash
    jupyter notebook price_prediction_without_outliers.ipynb
```

## 🧠 Technologies Used  
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook

## 📈 Results  

- The trained model provides price predictions with reasonable accuracy.  
- Removing outliers significantly improved performance and reduced prediction errors.
- **Selected Model**: Linear Regression
- **Final Poot Mean Squared Error(RMSE)**: 350505.7900560198
```
    Final_RSME.ipynb
```

## 📌 Future Work  

- Incorporate more features such as distance to metro, schools, hospitals, etc.  
- Deploy the model using Flask/Streamlit for real-time predictions.  
- Improve accuracy using advanced algorithms like XGBoost or Neural Networks.  

## Acknowledgements

 - [Statistical Learning With Python](https://www.youtube.com/watch?v=uFwbrdvrAJs&list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ&index=20)
 - [Hands-On-Machine-Learning with Sci-kit Learn](https://archive.org/details/handson-machine-learning-with-scikit-2-e-1)


## Documentation

[Project Report](unavailable)

