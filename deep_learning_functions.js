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

function calculateErrors(predictions, expectedResults) {
  const delta = vectorSubtract(predictions, expectedResults);

  return delta.map(number => number * number);
}

function dot(vector1, vector2) {
  if (vector1.length !== vector2.length)
    throw new Error("vectors must be same length");

  return vectorMultiply(vector1, vector2).reduce(
    (sum, number) => sum + number,
    0
  );
}

function vectorMatrixMultiply(vector, matrix) {
  if (vector.length !== matrix.length)
    throw new Error("vector and matrix must have the same length");
  return matrix.map(row => dot(vector, row));
}

function createMatrix(width, height) {
  const matrix = [];

  for (let _count = 0; _count < height; _count = _count + 1) {
    const row = [];
    for (_rowCount = 0; _rowCount < width; _rowCount = _rowCount + 1) {
      row.push(undefined);
    }
    matrix.push(row);
  }

  return matrix;
}

function outerProduct(inputs, deltas) {
  return deltas.map(delta => inputs.map(currentInput => currentInput * delta));
}

function scalarMatrixMultiply(scalar, matrix) {
  return matrix.map(matrixVector => scalarVectorMultiply(scalar, matrixVector));
}

function matrixSubtract(matrix1, matrix2) {
  return matrix1.map((matrix1Vector, index) =>
    vectorSubtract(matrix1Vector, matrix2[index])
  );
}

function createZerosMatrix(width, height) {
  const matrix = createMatrix(width, height);

  return matrix.map(row => row.map(_ => 0));
}

function calculateAccuracy(
  inputs,
  weights,
  expectedPredictions,
  neuralNetwork
) {
  let correctCount = 0;
  const totalRunCount = inputs.length;

  inputs.forEach((currentInput, index) => {
    const predictions = neuralNetwork(currentInput, weights); // [0.1, 0.1, 0.999999, 0.1]
    const predictionIndex = findLargestIndex(predictions);

    if (expectedPredictions[index][predictionIndex] === 1)
      correctCount = correctCount + 1;
  });

  return correctCount / totalRunCount;
}

function findLargestIndex(array) {
  return array.findIndex(currentValue => currentValue === Math.max(...array));
}

function createRandomMatrix(width, height) {
  const matrix = [];

  for (
    let _heightCount = 0;
    _heightCount < height;
    _heightCount = _heightCount + 1
  ) {
    const row = [];

    for (
      let _widthCount = 0;
      _widthCount < width;
      _widthCount = _widthCount + 1
    ) {
      row.push(Math.random() * 2 - 1);
    }

    matrix.push(row);
  }

  return matrix;
}

function dotVectorMatrix(vector, matrix) {
  return transpose(matrix).map(row => dot(vector, row));
}

function relu(vector) {
  return vector.map(item => (item > 0 ? item : 0));
}

function transpose(matrix) {
  const transposedMatrix = [];

  for (
    let matrixWidthIndex = 0;
    matrixWidthIndex < matrix[0].length;
    matrixWidthIndex = matrixWidthIndex + 1
  ) {
    const newRow = [];

    matrix.forEach(row => newRow.push(row[matrixWidthIndex]));

    transposedMatrix.push(newRow);
  }

  return transposedMatrix;
}

function reluToDerivative(vector) {
  return vector.map(item => (item > 0 ? 1 : 0));
}

module.exports = {
  vectorMultiply,
  scalarVectorMultiply,
  vectorSubtract,
  calculateErrors,
  dot,
  vectorMatrixMultiply,
  createMatrix,
  outerProduct,
  scalarMatrixMultiply,
  matrixSubtract,
  createZerosMatrix,
  findLargestIndex,
  calculateAccuracy,
  createRandomMatrix,
  dotVectorMatrix,
  relu,
  transpose,
  reluToDerivative
};
