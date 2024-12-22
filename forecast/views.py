import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dummy data
df = pd.read_csv('dummy_sales.csv')

# Train the model
X = df[['Advertising', 'Budget', 'Competition']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def home(request):
    return render(request, 'forecast/home.html')

def predict(request):
    if request.method == 'POST':
        # Get form data
        advertising = float(request.POST['advertising'])
        budget = float(request.POST['budget'])
        competition = float(request.POST['competition'])

        # Make prediction
        prediction = model.predict([[advertising, budget, competition]])[0]

        return render(request, 'forecast/result.html', {'prediction': prediction})
    else:
        return render(request, 'forecast/home.html')
