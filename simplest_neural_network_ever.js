function neuralNetwork(input, weights) {
  const prediction = vectorMatrixMultiplication(input, weights);

  return prediction;
}

function weightedSum(vector1, vector2) {
  const product = vectorMultiply(vector1, vector2);

  return product.reduce((sum, next) => sum + next, 0);
}

function vectorMultiply(vector1, vector2) {
  if (vector1.length !== vector2.length)
    throw new Error("vectors must be the same length");

  const resultVector = vector1.reduce((result, next, index) => {
    result.push(next * vector2[index]);
    return result;
  }, []);

  return resultVector;
}

function elementVectorMultipy(number, vector) {
  return vector.map(next => number * next);
}

function vectorMatrixMultiplication(vector, matrix) {
  if (vector.length !== matrix.length)
    throw new Error("vector and matrix must be the same length");

  return vector.map((_number, index) => weightedSum(vector, matrix[index]));
}

const number_of_fingers = [8.5, 10, 9, 10];
const win_loss_average = [0.65, 0.7, 0.6, 0.5];
const number_of_fouls = [2.5, 1, 1.5, 2];

const input = [number_of_fingers[0], win_loss_average[0], number_of_fouls[0]];
const weights = [
  //  finger, w/l, fouls
  [0.09, 0.9, 0.2], // win?
  [0.01, 0.1, 0.9], // break a bat?
  [0.1, 0.1, 0.9] // hit a fan?
];

const prediction = neuralNetwork(input, weights);

console.log(prediction);
