import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("data.csv")

print("--- 📊 TASK 1: DATA PROFILE ---")
print("Dataset Row and Column Count:", df.shape)
print("\nMissing Values Count per Column:")
print(df.isnull().sum())


df['Salary_LPA'] = df['Salary_LPA'].fillna(df['Salary_LPA'].median())
df['Company_Rating'] = df['Company_Rating'].fillna(df['Company_Rating'].median())
df['Openings'] = df['Openings'].fillna(1)
df['Applicants'] = df['Applicants'].fillna(0)
df = df.dropna(subset=['Job_Title', 'City', 'Experience_Level', 'Company_Type', 'Skills_Required'])
print("-" * 50)


print("\n--- 🏢 TASK 2: COMPANY TYPE ANALYSIS ---")
print(df.groupby('Company_Type')[['Salary_LPA', 'Applicants']].mean())
print("-" * 50)

print("\n--- 📍 TASK 3: CITY-WISE HIRING TRENDS ---")
print(pd.crosstab(df['City'], df['Location_Tier']))
print("-" * 50)

print("\n--- 🎓 TASK 4: FRESHER VS SENIOR SALARY MATRIX ---")
print(df.groupby(['Company_Type', 'Experience_Level'])['Salary_LPA'].mean().unstack())
print("-" * 50)



print("\n--- 🤖 TASK 5: SKILLS DEMAND ANALYSIS ---")
skill_salary_map = {}

for idx, row in df.iterrows():

    individual_skills = [skill.strip() for skill in str(row['Skills_Required']).split(',')]
    current_salary = row['Salary_LPA']
    
    for skill in individual_skills:
        if skill not in skill_salary_map:
            skill_salary_map[skill] = []
        skill_salary_map[skill].append(current_salary)


skill_avg_payout = {skill: np.mean(salaries) for skill, salaries in skill_salary_map.items()}
highest_paying_skills = sorted(skill_avg_payout.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Highest Paying Skills:")
for skill, avg_salary in highest_paying_skills:
    print(f" - {skill}: {avg_salary:.2f} LPA")
print("-" * 50)


print("\n--- 🧮 TASK 6: SALARY PREDICTION ENGINE ---")


X = df[['Job_Title', 'City', 'Experience_Level']].copy()
y = df['Salary_LPA']


le_title = LabelEncoder()
le_city = LabelEncoder()
le_exp = LabelEncoder()

X['Job_Title'] = le_title.fit_transform(X['Job_Title'])
X['City'] = le_city.fit_transform(X['City'])
X['Experience_Level'] = le_exp.fit_transform(X['Experience_Level'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, predictions)))
print("-" * 50)



plt.figure(figsize=(10, 6))

sns.barplot(
    data=df, 
    x='Company_Type', 
    y='Salary_LPA', 
    hue='Experience_Level', 
    errorbar=None
)

plt.title('Salary Compensation Matrix across Company Types')
plt.xlabel('Company Type')
plt.ylabel('Average Salary (LPA)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
