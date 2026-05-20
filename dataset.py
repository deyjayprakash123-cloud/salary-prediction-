import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data.csv")


df['Salary_LPA'] = df['Salary_LPA'].fillna(df['Salary_LPA'].median())


df = df.dropna(subset=['City', 'Experience_Level', 'Work_Mode', 'Salary_LPA'])


X = df[['City', 'Experience_Level', 'Work_Mode']].copy()
y = df['Salary_LPA']
le_city = LabelEncoder()
le_exp = LabelEncoder()
le_mode = LabelEncoder()


X['City'] = le_city.fit_transform(X['City'])
X['Experience_Level'] = le_exp.fit_transform(X['Experience_Level'])
X['Work_Mode'] = le_mode.fit_transform(X['Work_Mode'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Model trained successfully!")
print("Model Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2), "LPA")
print("-" * 50)



print("\n🔮 --- CUSTOM SALARY PREDICTOR SYSTEM --- 🔮")
print("Please enter the details below to predict expected salary:")


valid_cities = list(le_city.classes_)
valid_exps = list(le_exp.classes_)
valid_modes = list(le_mode.classes_)

print(f"\nAvailable Cities in dataset: {valid_cities}")
user_city = input("Enter City Name exactly as shown above: ")
print(f"Available Experience Levels: {valid_exps}")
user_exp = input("Enter Experience Level exactly as shown above: ")
print(f"Available Job Modes: {valid_modes}")
user_mode = input("Enter Job Mode exactly as shown above: ")


if (user_city in valid_cities) and (user_exp in valid_exps) and (user_mode in valid_modes):
    
    
    encoded_city = le_city.transform([user_city])[0]
    encoded_exp = le_exp.transform([user_exp])[0]
    encoded_mode = le_mode.transform([user_mode])[0]
    
  
    user_features = np.array([[encoded_city, encoded_exp, encoded_mode]])
    
  
    predicted_salary = model.predict(user_features)[0]
    
    print("\n" + "="*40)
    print(f"🚀 PREDICTED SALARY: {predicted_salary:.2f} LPA")
    print("="*40)

else:
    print("\n❌ Input Error: One or more inputs were spelled incorrectly. Please try again and match case exactly!")
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Work_Mode', y='Salary_LPA', hue='Experience_Level', errorbar=None)
plt.title('Salary Compensation Matrix across Job Modes & Experience')
plt.xlabel('Job Mode')
plt.ylabel('Average Salary (LPA)')
plt.tight_layout()
plt.show()
