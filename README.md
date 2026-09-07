Part B — Questions and Answers
Q1. What is Predictive Analytics, and how is it used in real-world applications?

Answer:

Predictive Analytics is the process of using historical data, statistical techniques, and machine learning algorithms to predict future outcomes.

It helps organizations make better decisions based on patterns present in previous data.

Real-world applications:
Business: Predicting future sales.
Banking: Predicting loan default risk.
Healthcare: Predicting disease risk.
E-commerce: Predicting customer purchases.
Real Estate: Predicting house prices.

Linear Regression is one of the commonly used supervised learning techniques for predicting continuous numerical values.

In simple words:

Predictive analytics = Using past data to predict what may happen in the future.

Q2. Explain the working principle of the Linear Regression algorithm.

Answer:

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

It tries to find the best relationship between independent variables (X) and a dependent variable (Y).

For simple Linear Regression, the equation is:

$$ Y = mX + c $$

Where:

Y = predicted output
X = input feature
m = slope/coefficient
c = intercept

For multiple Linear Regression:

$$ Y = b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n $$

The algorithm finds coefficients that minimize the difference between the actual values and predicted values.

Example:

Suppose we want to predict house price based on house area.

Area → Linear Regression Model → Predicted House Price

If area increases, the model may learn that house price generally increases.

Q3. Differentiate between dependent and independent variables with suitable examples.
Independent Variable	Dependent Variable
Input variable	Output variable
Used to make predictions	Value being predicted
Also called feature	Also called target
Represented by X	Represented by Y
Example:

Suppose we want to predict a student's marks based on study hours.

Study hours → Independent variable (X)
Marks → Dependent variable (Y)

Another example:

For house-price prediction:

House area
Number of bedrooms
Location-related features

can be independent variables, while house price is the dependent variable.

Q4. Why is it necessary to split the dataset into training and testing sets?

Answer:

We divide the dataset into training and testing sets to determine whether the model can make accurate predictions on data it has not seen before.

A common split is:

80% → Training data
20% → Testing data

The training data is used to learn the relationship between features and target values.

The testing data is then used to evaluate the trained model.

Example:
Dataset
   ↓
80% Training → Train Model
20% Testing  → Evaluate Model

The experiment specifically recommends an appropriate train-test ratio such as 80:20.

Main purpose: To check the model's ability to generalize to unseen data.

Q5. What is the significance of the R² Score in regression analysis?

Answer:

R² Score (R-squared) measures how well the independent variables explain the variation in the dependent variable.

Its general range is:

$$ -\infty \leq R^2 \leq 1 $$

A value closer to 1 generally indicates better explanatory performance.

Example:

If:

$$ R^2 = 0.80 $$

it means the model explains approximately 80% of the variation in the target variable on the evaluated data.

Interpretation:
R² Score	General Interpretation
1.0	Perfect fit
0.8	Strong fit
0.5	Moderate fit
0	No explanatory power relative to the baseline
< 0	Can be worse than the baseline

Important: A high R² alone does not prove that a model is appropriate; other metrics and assumptions should also be considered.

Q6. Differentiate between MAE, MSE, and RMSE. Which metric is more sensitive to large prediction errors?
1. MAE — Mean Absolute Error

MAE calculates the average absolute difference between actual and predicted values.

$$ MAE = \frac{1}{n}\sum |y_i-\hat{y_i}| $$

It is easy to understand because it is in the same unit as the target variable.

2. MSE — Mean Squared Error

MSE calculates the average of squared prediction errors.

$$ MSE = \frac{1}{n}\sum(y_i-\hat{y_i})^2 $$

Because errors are squared, large errors receive much greater weight.

3. RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

$$ RMSE = \sqrt{MSE} $$

It is also expressed in the same units as the target.

Comparison
Metric	Calculation	Large Error Sensitivity
MAE	Absolute error	Lower
MSE	Squared error	High
RMSE	√MSE	High
Answer:

MSE is especially sensitive to large prediction errors because the errors are squared. RMSE inherits this sensitivity.

The experiment specifically asks us to evaluate the model using MAE, MSE, RMSE, and R².

Q7. What assumptions should be satisfied before applying Linear Regression?

The important assumptions are:

1. Linearity

There should be an approximately linear relationship between predictors and the target.

2. Independence

Observations/errors should be reasonably independent.

3. Homoscedasticity

The variance of errors should remain approximately constant across predicted values.

4. Normality of residuals

For statistical inference, residuals are often assumed to be approximately normally distributed.

5. Low multicollinearity

Independent variables should not be excessively correlated with one another.

6. No extreme influential outliers

Extreme observations can strongly affect the fitted regression line.

In simple words:
Before using Linear Regression, the data should have a reasonably linear relationship and the model errors should behave appropriately.

Q8. How can overfitting and underfitting affect the performance of a regression model?
Overfitting

Overfitting occurs when a model learns the training data too closely, including noise.

Result:
Very good training performance
Poor testing performance
Poor generalization
Training Accuracy → High
Testing Accuracy  → Low
Underfitting

Underfitting occurs when the model is too simple to capture the important patterns in the data.

Result:
Poor training performance
Poor testing performance
Training Accuracy → Low
Testing Accuracy  → Low
Comparison
Overfitting	Underfitting
Model is too complex	Model is too simple
Learns noise	Misses important patterns
Training performance high	Training performance low
Testing performance poor	Testing performance poor

A good model should provide a reasonable balance between learning the data and generalizing to unseen data.

Q9. Mention any three real-world applications of Linear Regression in business or industry.

Three applications are:

1. House Price Prediction

Linear Regression can estimate house prices using features such as area, rooms, and other relevant characteristics.

2. Sales Forecasting

Businesses can use historical sales data and related variables to estimate future sales.

3. Advertising/Sales Analysis

A company can analyze the relationship between advertising expenditure and sales.

Other applications include:

Demand forecasting
Revenue prediction
Cost estimation
Economic forecasting

The experiment itself gives examples including House Price Prediction and Advertising Sales datasets.

Q10. How can feature selection improve the accuracy and interpretability of a predictive model?

Answer:

Feature selection means choosing the most relevant input variables for the model.

It can improve a predictive model by:

Removing irrelevant features
Reducing noise
Reducing computational complexity
Reducing the possibility of overfitting
Improving interpretability
Making important relationships easier to understand
Example:

Suppose house price prediction has 20 features, but only 6 are strongly relevant.
