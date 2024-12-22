
import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os

# Load the dummy dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, 'dummy_sales.csv'))

# Prepare the data
X = df[['Advertising', 'Budget', 'Competition']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Train models
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# Views
def home(request):
    return render(request, 'forecast/home.html')

def predict(request):
    if request.method == 'POST':
        # Get form data
        advertising = float(request.POST['advertising'])
        budget = float(request.POST['budget'])
        competition = float(request.POST['competition'])
        algorithm = request.POST['algorithm']

        # Get the selected model
        selected_model = trained_models.get(algorithm)

        # Make prediction
        if selected_model:
            prediction = selected_model.predict([[advertising, budget, competition]])[0]
            context = {
                'prediction': round(prediction, 2),
                'algorithm': algorithm,
            }
            return render(request, 'forecast/result.html', context)
        else:
            return render(request, 'forecast/home.html', {'error': 'Invalid Algorithm Selected'})
    else:
        return render(request, 'forecast/home.html')
