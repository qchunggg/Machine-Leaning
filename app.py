from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load các mô hình đã được huấn luyện
lin_reg_model = joblib.load('models/linear_regression_model.joblib')
lasso_model = joblib.load('models/lasso_regression_model.joblib')
nn_model = joblib.load('models/neural_network_model.joblib')
stacking_model = joblib.load('models/stacking_regressor_model.joblib')

# Load scaler đã được fit
scaler = joblib.load('models/scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    form_data = {}
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        form_data = {
            'year': int(request.form['year']),
            'present_price': float(request.form['present_price']),
            'kms_driven': int(request.form['kms_driven']),
            'fuel_type': int(request.form['fuel_type']),
            'seller_type': int(request.form['seller_type']),
            'transmission': int(request.form['transmission']),
            'owner': int(request.form['owner']),
            'model': request.form['model']
        }

        # Tạo DataFrame từ dữ liệu nhập vào
        input_data = pd.DataFrame([[form_data['year'], form_data['present_price'], form_data['kms_driven'], 
                                    form_data['fuel_type'], form_data['seller_type'], form_data['transmission'], form_data['owner']]],
                                  columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])

        # Chuẩn hóa dữ liệu đầu vào
        input_data_scaled = scaler.transform(input_data)

        # Dự đoán giá xe bằng mô hình được chọn
        if form_data['model'] == 'linear':
            prediction = round(lin_reg_model.predict(input_data)[0], 2)
        elif form_data['model'] == 'lasso':
            prediction = round(lasso_model.predict(input_data)[0], 2)
        elif form_data['model'] == 'neural_network':
            prediction = round(nn_model.predict(input_data_scaled)[0], 2)

    return render_template('index.html', prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
