from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

linear_model = joblib.load("./models/linear_regression_model.pkl")
rf_model = joblib.load("./models/random_forest_model.pkl")

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
  if request.method == 'POST':
    Area = float(request.form.get('Area'))
    Location = float(request.form.get('Location'))
    No_of_Bedrooms = float(request.form.get('No. of Bedrooms'))
    Resale = float(request.form.get('Resale'))
    MaintenanceStaff = float(request.form.get('MaintenanceStaff'))
    Gymnasium = float(request.form.get('Gymnasium'))
    SwimmingPool = float(request.form.get('SwimmingPool'))
    ShoppingMall = float(request.form.get('ShoppingMall'))
    SportsFacility = float(request.form.get('SportsFacility'))
    School = float(request.form.get('School'))
    _24X7Security = float(request.form.get('24X7Security'))
    CarParking = float(request.form.get('CarParking'))
    StaffQuarter = float(request.form.get('StaffQuarter'))
    Cafeteria = float(request.form.get('Cafeteria'))
    Hospital = float(request.form.get('Hospital'))
    LiftAvailable = float(request.form.get('LiftAvailable'))
    Furnished = float(request.form.get('Furnished'))
    
    new_data = [[Area, Location, No_of_Bedrooms, Resale, MaintenanceStaff, Gymnasium, SwimmingPool, ShoppingMall, SportsFacility, School, _24X7Security, CarParking, StaffQuarter, Cafeteria, Hospital, LiftAvailable, Furnished]]
    
    prediction_lr = linear_model.predict(new_data)
    prediction_rf = rf_model.predict(new_data)
    
    return render_template('index.html', result_lr=prediction_lr[0], result_rf=prediction_rf[0])
    
  else:
   return render_template('predict.html')

if __name__ == "__main__":
  app.run(debug=True)