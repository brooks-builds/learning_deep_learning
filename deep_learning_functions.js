function vectorMultiply(vector1, vector2) {
  if (vector1.length !== vector2.length)
    throw new Error("vectors must be same length");
  return vector1.map((firstNumber, index) => firstNumber * vector2[index]);
}

function scalarVectorMultiply(scalar, vector) {
  return vector.map(item => item * scalar);
}

function vectorSubtract(vector1, vector2) {
  if (vector1.length !== vector2.length)
    throw new Error("vectors must be same length");
  return vector1.map((firstNumber, index) => firstNumber - vector2[index]);
}

function calculateErrors(vector1, vector2) {
  const delta = vectorSubtract(vector1, vector2);

  return delta.map(number => number * number);
}

module.exports = {
  vectorMultiply,
  scalarVectorMultiply,
  vectorSubtract,
  calculateErrors
};
