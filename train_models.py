# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor  # Mô hình Neural Network
import os
from sklearn.model_selection import GridSearchCV

# Create 'graph' directory if it doesn't exist
if not os.path.exists('graph'):
    os.makedirs('graph')

# Thu thập và xử lý dữ liệu
# Đọc dữ liệu từ file CSV vào dataframe của pandas
car_dataset = pd.read_csv('./CarData.csv')

# Kiểm tra 5 dòng đầu tiên của dữ liệu
print("5 dòng đầu tiên của dữ liệu:")
print(car_dataset.head())

# Kiểm tra số hàng và cột
print("\nSố hàng và cột:")
print(car_dataset.shape)

# Thông tin về dữ liệu
print("\nThông tin về dữ liệu:")
print(car_dataset.info())

# Kiểm tra kiểu dữ liệu của các cột
print("\nKiểu dữ liệu của các cột:")
print(car_dataset.dtypes)

# Kiểm tra giá trị bị thiếu
print("\nSố lượng giá trị bị thiếu trong mỗi cột:")
print(car_dataset.isnull().sum())

# Kiểm tra tỷ lệ phần trăm giá trị bị thiếu
print("\nTỷ lệ phần trăm giá trị bị thiếu trong mỗi cột:")
print((car_dataset.isnull().sum() / len(car_dataset)) * 100)

# Kiểm tra giá trị trùng lặp
duplicates = car_dataset.duplicated()
print("\nSố lượng hàng trùng lặp:")
print(duplicates.sum())

if duplicates.sum() > 0:
    print("\nCác hàng trùng lặp:")
    print(car_dataset[duplicates])

# Kiểm tra phân phối của các dữ liệu dạng danh mục (categorical)
print("\nPhân phối của dữ liệu dạng danh mục:")
print("Fuel_Type:")
print(car_dataset.Fuel_Type.value_counts())
print("\nSeller_Type:")
print(car_dataset.Seller_Type.value_counts())
print("\nTransmission:")
print(car_dataset.Transmission.value_counts())

# Thống kê mô tả cho các cột số
print("\nThống kê mô tả cho các cột số:")
print(car_dataset.describe())

# Kiểm tra giá trị ngoại lai và xóa chúng
numeric_columns = car_dataset.select_dtypes(include=[np.number]).columns

print("\nKiểm tra và xóa giá trị ngoại lai:")
total_outliers = 0
for column in numeric_columns:
    Q1 = car_dataset[column].quantile(0.25)
    Q3 = car_dataset[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = car_dataset[(car_dataset[column] < lower_bound) | (car_dataset[column] > upper_bound)]
    
    print(f"\nCột {column}:")
    print(f"Số lượng giá trị ngoại lai: {len(outliers)}")
    print(f"Phần trăm giá trị ngoại lai: {(len(outliers) / len(car_dataset)) * 100:.2f}%")
    print(f"Giới hạn dưới: {lower_bound}")
    print(f"Giới hạn trên: {upper_bound}")
    
    if len(outliers) > 0:
        print("Các giá trị ngoại lai:")
        print(outliers[column].tolist())
        
    # Xóa các giá trị ngoại lai
    car_dataset = car_dataset[(car_dataset[column] >= lower_bound) & (car_dataset[column] <= upper_bound)]
    total_outliers += len(outliers)

print(f"\nTổng số giá trị ngoại lai đã xóa: {total_outliers}")
print(f"Số lượng mẫu còn lại sau khi xóa giá trị ngoại lai: {len(car_dataset)}")

# Mã hóa dữ liệu dạng danh mục
# Mã hóa cột "Fuel_Type"
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# Mã hóa cột "Seller_Type"
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)

# Mã hóa cột "Transmission"
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

print(car_dataset.head())

# Tách dữ liệu và nhãn
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Tách tập dữ liệu thành tập huấn luyện, tập kiểm tra và tập xác thực
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=2)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=2)

# Chuẩn hóa dữ liệu (chỉ cần cho mô hình mạng nơron)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

### 1. Mô hình Linear Regression (Hồi quy tuyến tính)

# Tạo mô hình Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện, kiểm tra và xác thực
train_predictions_lin = lin_reg_model.predict(X_train)
test_predictions_lin = lin_reg_model.predict(X_test)
val_predictions_lin = lin_reg_model.predict(X_val)

# Tính toán các chỉ số đánh giá cho Linear Regression
train_r2_score_lin = r2_score(Y_train, train_predictions_lin)
test_r2_score_lin = r2_score(Y_test, test_predictions_lin)
val_r2_score_lin = r2_score(Y_val, val_predictions_lin)
train_mse_lin = mean_squared_error(Y_train, train_predictions_lin)
test_mse_lin = mean_squared_error(Y_test, test_predictions_lin)
val_mse_lin = mean_squared_error(Y_val, val_predictions_lin)
train_rmse_lin = np.sqrt(train_mse_lin)
test_rmse_lin = np.sqrt(test_mse_lin)
val_rmse_lin = np.sqrt(val_mse_lin)

print(
    f"Linear Regression - R² trên tập huấn luyện: {train_r2_score_lin:.4f}, R² trên tập kiểm tra: {test_r2_score_lin:.4f}, R² trên tập xác thực: {val_r2_score_lin:.4f}")
print(
    f"Linear Regression - MSE trên tập huấn luyện: {train_mse_lin:.4f}, MSE trên tập kiểm tra: {test_mse_lin:.4f}, MSE trên tập xác thực: {val_mse_lin:.4f}")
print(
    f"Linear Regression - RMSE trên tập huấn luyện: {train_rmse_lin:.4f}, RMSE trên tập kiểm tra: {test_rmse_lin:.4f}, RMSE trên tập xác thực: {val_rmse_lin:.4f}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions_lin, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Linear Regression: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions_lin, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Linear Regression: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions_lin, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Linear Regression: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/linear_regression.png')  # Lưu biểu đồ
plt.close()

### 2. Mô hình Lasso Regression

# Tạo mô hình Lasso Regression với các giá trị alpha khác nhau
alphas = np.logspace(-4, 4, 50)
best_alpha = None
best_r2_score = -np.inf

# Tìm alpha tốt nhất
for alpha in alphas:
    lass_reg_model = Lasso(alpha=alpha)
    lass_reg_model.fit(X_train, Y_train)

    # Dự đoán trên tập kiểm tra
    test_predictions_lasso = lass_reg_model.predict(X_test)

    # Tính toán chỉ số R² trên tập kiểm tra
    test_r2_score_lasso = r2_score(Y_test, test_predictions_lasso)

    # Lưu lại alpha tốt nhất
    if test_r2_score_lasso > best_r2_score:
        best_r2_score = test_r2_score_lasso
        best_alpha = alpha  # Cập nhật best_alpha

# Lấy 4 chữ số sau dấu phẩy cho alpha tốt nhất
best_alpha_rounded = round(best_alpha, 4)

# In ra alpha tốt nhất
print(f"Alpha tốt nhất cho Lasso Regression: {best_alpha_rounded}")

# Tạo mô hình Lasso Regression với alpha tốt nhất
best_lass_reg_model = Lasso(alpha=best_alpha)
best_lass_reg_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện, kiểm tra và xác thực
train_predictions_lasso = best_lass_reg_model.predict(X_train)
test_predictions_lasso = best_lass_reg_model.predict(X_test)
val_predictions_lasso = best_lass_reg_model.predict(X_val)

# Tính toán các chỉ số đánh giá cho Lasso Regression
train_r2_score_lasso = r2_score(Y_train, train_predictions_lasso)
test_r2_score_lasso = r2_score(Y_test, test_predictions_lasso)
val_r2_score_lasso = r2_score(Y_val, val_predictions_lasso)
train_mse_lasso = mean_squared_error(Y_train, train_predictions_lasso)
test_mse_lasso = mean_squared_error(Y_test, test_predictions_lasso)
val_mse_lasso = mean_squared_error(Y_val, val_predictions_lasso)
train_rmse_lasso = np.sqrt(train_mse_lasso)
test_rmse_lasso = np.sqrt(test_mse_lasso)
val_rmse_lasso = np.sqrt(val_mse_lasso)

print(
    f"Lasso Regression - R² trên tập huấn luyện: {train_r2_score_lasso:.4f}, R² trên tập kiểm tra: {test_r2_score_lasso:.4f}, R² trên tập xác thực: {val_r2_score_lasso:.4f}")
print(
    f"Lasso Regression - MSE trên tập huấn luyện: {train_mse_lasso:.4f}, MSE trên tập kiểm tra: {test_mse_lasso:.4f}, MSE trên tập xác thực: {val_mse_lasso:.4f}")
print(
    f"Lasso Regression - RMSE trên tập huấn luyện: {train_rmse_lasso:.4f}, RMSE trên tập kiểm tra: {test_rmse_lasso:.4f}, RMSE trên tập xác thực: {val_rmse_lasso:.4f}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions_lasso, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Lasso Regression: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions_lasso, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Lasso Regression: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions_lasso, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Lasso Regression: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/lasso_regression.png')  # Lưu biểu đồ
plt.close()

### 3. Mô hình Neural Network (Mạng Nơron)

# Đảm bảo dữ liệu đã được chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Định nghĩa mô hình Neural Network
nn_model = MLPRegressor(random_state=42, max_iter=500)

# Định nghĩa grid các hyperparameters cần tìm kiếm
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [32, 64, 128]
}

# Sử dụng GridSearchCV
grid_search = GridSearchCV(nn_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, Y_train)

# In ra best parameters và best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Lấy mô hình tốt nhất
best_nn_model = grid_search.best_estimator_

# Đánh giá mô hình
train_predictions = best_nn_model.predict(X_train_scaled)
test_predictions = best_nn_model.predict(X_test_scaled)
val_predictions = best_nn_model.predict(X_val_scaled)

train_r2 = r2_score(Y_train, train_predictions)
test_r2 = r2_score(Y_test, test_predictions)
val_r2 = r2_score(Y_val, val_predictions)

print(f"Neural Network - R² trên tập huấn luyện: {train_r2:.4f}, R² trên tập kiểm tra: {test_r2:.4f}, R² trên tập xác thực: {val_r2:.4f}")

# Vẽ đồ thị loss function
plt.figure(figsize=(10, 6))
plt.plot(best_nn_model.loss_curve_)
plt.title('Loss Curve của Neural Network')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('graph/neural_network_loss.png')
plt.close()

# In ra số lượng iterations thực tế
print(f"Số lượng iterations thực tế: {best_nn_model.n_iter_}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Neural Network: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Neural Network: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Neural Network: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/neural_network.png')  # Lưu biểu đồ
plt.close()

### 4. Stacking Regressor

# Định nghĩa mô hình stacking regressor
base_models = [
    ('linear', lin_reg_model),
    ('lasso', best_lass_reg_model),
    ('nn', best_nn_model)  # Use the best Neural Network model
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
stacking_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện, kiểm tra và xác thực
train_predictions_stacked = stacking_model.predict(X_train)
test_predictions_stacked = stacking_model.predict(X_test)
val_predictions_stacked = stacking_model.predict(X_val)

# Tính toán các chỉ số đánh giá cho Stacking Regressor
train_r2_score_stacked = r2_score(Y_train, train_predictions_stacked)
test_r2_score_stacked = r2_score(Y_test, test_predictions_stacked)
val_r2_score_stacked = r2_score(Y_val, val_predictions_stacked)
train_mse_stacked = mean_squared_error(Y_train, train_predictions_stacked)
test_mse_stacked = mean_squared_error(Y_test, test_predictions_stacked)
val_mse_stacked = mean_squared_error(Y_val, val_predictions_stacked)
train_rmse_stacked = np.sqrt(train_mse_stacked)
test_rmse_stacked = np.sqrt(test_mse_stacked)
val_rmse_stacked = np.sqrt(val_mse_stacked)

print(
    f"Linear Regression - R² trên tập huấn luyện: {train_r2_score_lin:.4f}, R² trên tập kiểm tra: {test_r2_score_lin:.4f}, R² trên tập xác thực: {val_r2_score_lin:.4f}")
print(
    f"Linear Regression - MSE trên tập huấn luyện: {train_mse_lin:.4f}, MSE trên tập kiểm tra: {test_mse_lin:.4f}, MSE trên tập xác thực: {val_mse_lin:.4f}")
print(
    f"Linear Regression - RMSE trên tập huấn luyện: {train_rmse_lin:.4f}, RMSE trên tập kiểm tra: {test_rmse_lin:.4f}, RMSE trên tập xác thực: {val_rmse_lin:.4f}")

print(
    f"Lasso Regression - R² trên tập huấn luyện: {train_r2_score_lasso:.4f}, R² trên tập kiểm tra: {test_r2_score_lasso:.4f}, R² trên tập xác thực: {val_r2_score_lasso:.4f}")
print(
    f"Lasso Regression - MSE trên tập huấn luyện: {train_mse_lasso:.4f}, MSE trên tập kiểm tra: {test_mse_lasso:.4f}, MSE trên tập xác thực: {val_mse_lasso:.4f}")
print(
    f"Lasso Regression - RMSE trên tập huấn luyện: {train_rmse_lasso:.4f}, RMSE trên tập kiểm tra: {test_rmse_lasso:.4f}, RMSE trên tập xác thực: {val_rmse_lasso:.4f}")

print(
    f"Neural Network - R² trên tập huấn luyện: {train_r2:.4f}, R² trên tập kiểm tra: {test_r2:.4f}, R² trên tập xác thực: {val_r2:.4f}")
print(
    f"Neural Network - MSE trên tập huấn luyện: {train_mse_stacked:.4f}, MSE trên tập kiểm tra: {test_mse_stacked:.4f}, MSE trên tập xác thực: {val_mse_stacked:.4f}")
print(
    f"Neural Network - RMSE trên tập huấn luyện: {train_rmse_stacked:.4f}, RMSE trên tập kiểm tra: {test_rmse_stacked:.4f}, RMSE trên tập xác thực: {val_rmse_stacked:.4f}")

print(
    f"Stacking Regressor - R² trên tập huấn luyện: {train_r2_score_stacked:.4f}, R² trên tập kiểm tra: {test_r2_score_stacked:.4f}, R² trên tập xác thực: {val_r2_score_stacked:.4f}")
print(
    f"Stacking Regressor - MSE trên tập huấn luyện: {train_mse_stacked:.4f}, MSE trên tập kiểm tra: {test_mse_stacked:.4f}, MSE trên tập xác thực: {val_mse_stacked:.4f}")
print(
    f"Stacking Regressor - RMSE trên tập huấn luyện: {train_rmse_stacked:.4f}, RMSE trên tập kiểm tra: {test_rmse_stacked:.4f}, RMSE trên tập xác thực: {val_rmse_stacked:.4f}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions_stacked, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Stacking Regressor: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions_stacked, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Stacking Regressor: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions_stacked, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Stacking Regressor: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/stacking_regressor.png')  # Lưu biểu đồ
plt.close()

# Lưu các mô hình
import joblib
import os

# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(lin_reg_model, 'models/linear_regression_model.joblib')
joblib.dump(best_lass_reg_model, 'models/lasso_regression_model.joblib')
joblib.dump(best_nn_model, 'models/neural_network_model.joblib')
joblib.dump(stacking_model, 'models/stacking_regressor_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

print("All models have been saved in the 'models' directory.")