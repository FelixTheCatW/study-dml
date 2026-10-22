import numpy as np

def dense_layer(X, W, b):
    return X @ W + b

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def mlp_predict(X, W1, b1, W2, b2):
    Z1 = dense_layer(X, W1, b1)
    A1 = relu(Z1)
    Z2 = dense_layer(A1, W2, b2)
    return sigmoid(Z2)

def count_parameters(input_size, hidden_size, output_size):
    input_to_hidden = input_size * hidden_size
    hidden_bias = hidden_size
    hidden_to_output = hidden_size * output_size
    output_bias = output_size
    return input_to_hidden + hidden_bias + hidden_to_output + output_bias

#==================== dense_layer ====================
# 1) Скалярный/векторный тест: точные значения
X = np.array([[1.0, 2.0]])
W = np.array([[0.5, -1.0], [2.0, 1.0]])
b = np.array([0.1, 0.2])
# Считаем вручную: (1*0.5 + 2*2.0) + 0.1 = 4.6 ; (1*-1.0 + 2*1.0) + 0.2 = 1.2
assert np.allclose(dense_layer(X, W, b), np.array([[4.6, 1.2]]))

#====================  ====================
# 2) Батч: проверяем форму и независимость вычислений для каждой строки
X = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
W = np.array([[1.0, 0.0], [0.0, 1.0]])  # единичная матрица (2, 2)
b = np.array([10.0, 20.0])
# Z = X @ I + b -> просто X + b
assert dense_layer(X, W, b).shape == (2, 2)
assert np.allclose(dense_layer(X, W, b), np.array([[11.0, 22.0], [13.0, 24.0]]))

#==================== relu ====================
# 1) Отрицательные → 0, положительные — без изменений
assert np.array_equal(relu(np.array([-2.5, -0.1, 0.0, 1.5, 3.0])),
                      np.array([0.0, 0.0, 0.0, 1.5, 3.0]))

# 2) Свойство: выход всегда неотрицательный, и ReLU(x) >= x для любого x
z = np.linspace(-10, 10, 100)
assert np.all(relu(z) >= 0) and np.all(relu(z) >= z)

#==================== sigmoid ====================
# 1) Сигмоида от нуля всегда ровно 0.5
assert np.isclose(sigmoid(0), 0.5)

# 2) Свойство симметрии: σ(z) + σ(-z) = 1, и выход всегда в (0, 1)
z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
assert np.allclose(sigmoid(z) + sigmoid(-z), np.ones_like(z))
assert np.all((sigmoid(z) > 0) & (sigmoid(z) < 1))

#==================== mlp_predict ====================
# 1) Проверка формы и диапазона: выход — это вероятность (0, 1)
X_dummy = np.random.rand(5, 3)      # 5 студентов, 3 признака
W1, b1 = np.ones((3, 4)), np.zeros(4)    # (3, 4)
W2, b2 = np.ones((4, 1)), np.zeros(1)    # (4, 1)
preds = mlp_predict(X_dummy, W1, b1, W2, b2)
assert preds.shape == (5, 1)
assert np.all((preds > 0) & (preds < 1))

# 2) Тест с нулевыми весами: все Z = 0, sigmoid(0) = 0.5
X_zero = np.array([[1.0, 2.0]])
W1_zero, b1_zero = np.zeros((2, 3)), np.zeros(3)
W2_zero, b2_zero = np.zeros((3, 1)), np.zeros(1)
assert np.isclose(mlp_predict(X_zero, W1_zero, b1_zero, W2_zero, b2_zero)[0, 0], 0.5)

#==================== count_parameters ====================
# 1) Точный расчёт для архитектуры (3 -> 4 -> 1):
# Скрытый слой: 3*4 (веса) + 4 (смещения) = 16
# Выходной слой: 4*1 (веса) + 1 (смещение) = 5
# Итого: 21
assert count_parameters(3, 4, 1) == 21

# 2) Проверка формулы на других числах (2 -> 5 -> 3):
# 2*5 + 5 + 5*3 + 3 = 10 + 5 + 15 + 3 = 33
assert count_parameters(2, 5, 3) == 33
assert count_parameters(0, 0, 0) == 0  # Вырожденный случай


print("All tests passed")