import numpy as np

def weighted_sum(x, w, b):
    return np.dot(x, w) + b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def neuron_prediction(x, w, b):
    z = weighted_sum(x, w, b)
    return sigmoid(z)

#============== weighted_sum ===============
# 1) скалярный случай: 1*3 + 2*4 + 5 = 16
assert np.isclose(weighted_sum(np.array([1, 2]), np.array([3, 4]), 5), 16.0)

# 2) батч из двух примеров: строки считаются независимо
x = np.array([[1, 2],
              [0, 1]])
w = np.array([2, 3])
b = 1.0
# [1*2+2*3+1, 0*2+1*3+1] = [9, 4]
assert np.allclose(weighted_sum(x, w, b), np.array([9.0, 4.0]))

#============== sigmoid ===============
# 1) сигмоида от нуля всегда ровно 0.5
assert np.isclose(sigmoid(0), 0.5)

# 2) свойство симметрии: σ(z) + σ(−z) = 1
z = np.array([-2.0, -0.5, 0.5, 2.0])
assert np.allclose(sigmoid(z) + sigmoid(-z), np.ones_like(z))

#============== relu ===============
# 1) отрицательные → 0, положительные — без изменений
assert np.array_equal(relu(np.array([-3, -1, 0, 2, 5])), np.array([0, 0, 0, 2, 5]))

# 2) для скаляров: relu(-4) == 0 и relu(4) == 4
assert relu(-4) == 0 and relu(4) == 4

#============== neuron_prediction ===============
# 1) нулевые входы и веса, b = 0 → z = 0 → σ(0) = 0.5
assert np.isclose(neuron_prediction(np.array([0, 0]), np.array([1, 1]), 0), 0.5)

# 2) сильно положительный сдвиг b → выход стремится к 1
pred = neuron_prediction(np.array([0, 0]), np.array([1, 1]), 50.0)
assert pred > 0.999

print("All asserts passed")