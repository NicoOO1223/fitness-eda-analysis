import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_log_error
import os

# Create assets folder
if not os.path.exists('assets'):
    os.makedirs('assets')

# Load Data
try:
    df = pd.read_csv('data/train.csv')
except FileNotFoundError:
    df = pd.read_csv('train.csv')

# ---- Data Preparation ----
print("--- ⚙️ Preparing Data ---")
df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()

# Sample for visualization speed
df_sample = df.sample(10000, random_state=42)

# ---- Visualization 1: Correlation Matrix ----
print("--- 📊 Generating Correlation Matrix ---")
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/correlation_matrix.png')
plt.close()

# ---- Visualization 2: Multivariate Analysis ----
print("--- 📊 Generating Multivariate Analysis ---")
fig, ax = plt.subplots(1,2, figsize=(16,6))

# Duration vs Calories (Sex)
sns.scatterplot(x='Duration', y='Calories', data=df_sample, hue='Sex', alpha=0.6, ax=ax[0])
ax[0].set_title("Calories by Duration & Gender", fontsize=14, fontweight='bold')

# Bubble Chart (Heart Rate & Temp)
sns.scatterplot(
    data=df_sample, x='Duration', y='Calories', hue='Heart_Rate', 
    size='Body_Temp', palette='rocket_r', sizes=(30, 250), alpha=0.7, ax=ax[1]
)
ax[1].set_title('Impact of Heart Rate & Temp', fontsize=14, fontweight='bold')
sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('assets/multivariate_analysis.png')
plt.close()

# ---- Visualization 3: Distributions ----
print("--- 📊 Generating Distributions ---")
fig, axes = plt.subplots(1,3, figsize=(18,5))
sns.histplot(df['BMI'], kde=True, bins=30, ax=axes[0], color='skyblue')
axes[0].set_title("BMI Distribution", fontsize=14, fontweight='bold')

sns.histplot(df['Calories'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title("Calories Distribution", fontsize=14, fontweight='bold')

sns.violinplot(data=df_sample, x='Sex', y='Calories', ax=axes[2], palette='pastel')
axes[2].set_title("Calories by Sex", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('assets/distributions.png')
plt.close()

# ---- Modeling (XGBoost) ----
print("--- 🤖 Training XGBoost Model ---")
X = df.drop(columns=['id', 'Calories'])
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, 
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"✅ Model Results -> R2 Score: {r2:.4f}")

# ---- Visualization 4: Model Performance ----
print("--- 📊 Generating Performance Plots ---")
fig, ax = plt.subplots(1,2, figsize=(16,6))

# Residuals
residuals = y_test - y_pred
ax[0].scatter(y_pred, residuals, color='teal', alpha=0.6, edgecolor='black')
ax[0].axhline(0, color='red', linestyle='--')
ax[0].set_title("Residual Plot", fontsize=14, fontweight='bold')
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Residuals")

# Actual vs Predicted
ax[1].scatter(y_test, y_pred, color='purple', alpha=0.6, edgecolor='black')
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'orange', lw=2, linestyle='--')
ax[1].set_title(f"Actual vs Predicted (R2: {r2:.2f})", fontsize=14, fontweight='bold')
ax[1].set_xlabel("Actual")
ax[1].set_ylabel("Predicted")

plt.tight_layout()
plt.savefig('assets/model_performance.png')
plt.close()

print("\n🎉 Done! Charts saved to 'assets/' folder.")