import numpy


def relu(x):
    return (x > 0) * x  # returns 0 if x is less than 0


alpha = 0.2
hidden_size = 4

streetlights = numpy.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])

did_walk_vs_stop = numpy.array([1, 1, 0, 0])

# create matrix (3 height, hidden size width) with each value between -1 and 1
weights_0_1 = 2 * numpy.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * numpy.random.random((hidden_size, 1)) - 1

layer_0 = streetlights[0]
# dot the vector and matrix, and then change all negative numbers to 0
layer_1 = relu(numpy.dot(layer_0, weights_0_1))
# print(numpy.dot(layer_0, weights_0_1), layer_1)
layer_2 = numpy.dot(layer_1, weights_1_2)

# 1*1 + 2 * 4 + 3 * 7 = 30
# 1*2 + 2*5 + 3*8 = 36
# 1*3 + 2*6 + 3*9 = 42
# 1*4 + 2*7 + 3*10 = 48
# print(numpy.dot([1, 2, 3], [
#     [1, 2, 3, 4],
#     [4, 5, 6, 7],
#     [7, 8, 9, 10]
# ]))
