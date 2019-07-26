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
    matrix.push(new Array(width));
  }

  return matrix;
}

function outerProduct(vector1, vector2) {
  const matrix = createMatrix(vector1.length, vector2.length);

  for (
    let matrixHeightIndex = 0;
    matrixHeightIndex < vector2.length;
    matrixHeightIndex = matrixHeightIndex + 1
  ) {
    for (
      let matrixWidthIndex = 0;
      matrixWidthIndex < vector1.length;
      matrixWidthIndex = matrixWidthIndex + 1
    ) {
      matrix[matrixHeightIndex][matrixWidthIndex] =
        vector1[matrixHeightIndex] * vector2[matrixWidthIndex];
    }
  }

  return matrix;
}

function scalarMatrixMultiply(scalar, matrix) {
  return matrix.map(matrixVector => scalarVectorMultiply(scalar, matrixVector));
}

function matrixSubtract(matrix1, matrix2) {
  return matrix1.map((matrix1Vector, index) =>
    vectorSubtract(matrix1Vector, matrix2[index])
  );
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
  matrixSubtract
};
