import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_excel('Dataset.xlsx')


# Encoding the target variable
label_encoder = LabelEncoder()
df['AQI_Range_encoded'] = label_encoder.fit_transform(df['AQI_Range'])

# Splitting the dataset into features (X) and target variable (Y)
X = df[['soi', 'noi', 'SPMi']]
Y = df['AQI_Range_encoded']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=70)

# Training the Decision Tree Regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, Y_train)

# Making predictions
Y_pred = regressor.predict(X_test)

# Calculating Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print('Mean Squared Error:', mse)

# Save the model
joblib.dump(regressor, 'model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')