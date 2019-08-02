import numpy

numpy.random.seed(1)


def relu(x):
    return (x > 0) * x  # returns 0 if x is less than 0


def relu2deriv(output):
    return output > 0  # return True (1) if output is greater than 0


alpha = 0.01
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


# 1*1 + 2 * 4 + 3 * 7 = 30
# 1*2 + 2*5 + 3*8 = 36
# 1*3 + 2*6 + 3*9 = 42
# 1*4 + 2*7 + 3*10 = 48
# print(numpy.dot([1, 2, 3], [
#     [1, 2, 3, 4],
#     [4, 5, 6, 7],
#     [7, 8, 9, 10]
# ]))

for count in range(60):
    layer_2_error = 0

    # can instead do for (index, streetlight) in enumerate(streetlights):
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]  # returns an array of 1 value
        # dot the vector and matrix, and then change all negative numbers to 0
        layer_1 = relu(numpy.dot(layer_0, weights_0_1))
        # print(numpy.dot(layer_0, weights_0_1), layer_1)
        layer_2 = numpy.dot(layer_1, weights_1_2)

        # print('prediction', layer_2)
        # print('expected outcome', did_walk_vs_stop[i])

        layer_2_error += numpy.sum((layer_2 - did_walk_vs_stop[i:i+1]) ** 2)

        layer_2_delta = layer_2 - did_walk_vs_stop[i:i+1]
        # dot together layer 2 delta and a transposed layer 2 weights and then multiply the result with layer 1 prediction but with its values turned into 1 or 0 depending on if they are positive (1) or negative (0)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

        # add to weights the result of alpha being multiplied by the transposed layer 1 dotted with the layer 2 delta
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if(count % 10 == 0):
        print('error: ', layer_2_error)
