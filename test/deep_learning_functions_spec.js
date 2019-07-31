const chai = require("chai");

const {
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
  calculateAccuracy
} = require("../deep_learning_functions");

const assert = chai.assert;

describe("canary test", () => {
  it("5 should be 5", () => {
    assert.equal(5, 5);
  });
});

describe("vectorMultiply", () => {
  it("should multiply 2 vectors together", () => {
    const vectorA = [1, 2, 3];
    const vectorB = [4, 5, 6];
    const expectedOutput = [4, 10, 18];

    assert.deepEqual(vectorMultiply(vectorA, vectorB), expectedOutput);
  });

  it("should fail if 2 vectors of different sizes are passed in", () => {
    const vectorA = [1];
    const vectorB = [1, 2];

    assert.throws(
      vectorMultiply.bind(this, vectorA, vectorB),
      "vectors must be same length"
    );
  });
});

describe("scalarVectorMultiply", () => {
  it("should multiply a vector with a scalar", () => {
    const scalar = 10;
    const vector = [1, 2, 3];
    const expectedOutput = [10, 20, 30];

    assert.deepEqual(scalarVectorMultiply(scalar, vector), expectedOutput);
  });
});

describe("vectorSubtract", () => {
  it("should subtract one vector from the other", () => {
    const vector1 = [10, 20, 30];
    const vector2 = [5, 10, 15];
    const expectedOutput = [5, 10, 15];

    assert.deepEqual(vectorSubtract(vector1, vector2), expectedOutput);
  });
});

describe("calculateError", () => {
  it("should return the error vectors", () => {
    const vector1 = [10, 20, 30];
    const vector2 = [5, 10, 15];
    const expectedOutput = [25, 100, 225];

    assert.deepEqual(calculateErrors(vector1, vector2), expectedOutput);
  });
});

describe("vectorMatrixMultiply", () => {
  it("should return a vector", () => {
    const vector = [1, 2, 3];
    const matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const expectedOutput = [14, 32, 50];

    assert.deepEqual(vectorMatrixMultiply(vector, matrix), expectedOutput);
  });
});

describe("dot", () => {
  it("should dot (weighted sum / inner product) two vectors together", () => {
    const vector1 = [1, 2, 3];
    const vector2 = [4, 5, 6];
    const expectedOutput = 32;

    assert.deepEqual(dot(vector1, vector2), expectedOutput);
  });
});

describe("outerProduct", () => {
  it("should turn two vectors into a matrix", () => {
    const inputs = [1, 2, 3];
    const deltas = [4, 5, 6];
    const expectedOutput = [[4, 8, 12], [5, 10, 15], [6, 12, 18]];

    assert.deepEqual(outerProduct(inputs, deltas), expectedOutput);
  });

  it("should return a matrix that was created from vectors of different sizes", () => {
    const inputs = [1, 2, 3, 4, 5];
    const deltas = [6, 7, 8];
    const expectedOutput = [
      //input 1, 2, 3, 4, 5
      [6, 12, 18, 24, 30], // weight 1
      [7, 14, 21, 28, 35], // weight 2
      [8, 16, 24, 32, 40] // weight 3
    ];

    assert.deepEqual(outerProduct(inputs, deltas), expectedOutput);
  });
});

describe("createMatrix", () => {
  it("should create a matrix", () => {
    const width = 5;
    const height = 3;
    const expectedOutput = [
      [undefined, undefined, undefined, undefined, undefined],
      [undefined, undefined, undefined, undefined, undefined],
      [undefined, undefined, undefined, undefined, undefined]
    ];

    assert.deepEqual(createMatrix(width, height), expectedOutput);
  });

  it("should create a matrix with values set to 0", () => {
    const width = 5;
    const height = 3;
    const expectedOutput = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];

    assert.deepEqual(createZerosMatrix(width, height), expectedOutput);
  });
});

describe("scalarMatrixMultiply", () => {
  it("should multiply a scalar by a matrix and return a matrix", () => {
    const scalar = 10;
    const matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const expectedOutput = [[10, 20, 30], [40, 50, 60], [70, 80, 90]];

    assert.deepEqual(scalarMatrixMultiply(scalar, matrix), expectedOutput);
  });
});

describe("matrixSubtract", () => {
  it("shuld subtract one matrix from another, returning a matrix", () => {
    const matrix1 = [[10, 20, 30], [40, 50, 60], [70, 80, 90]];
    const matrix2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const expectedOutput = [[9, 18, 27], [36, 45, 54], [63, 72, 81]];

    assert.deepEqual(matrixSubtract(matrix1, matrix2), expectedOutput);
  });
});

describe("calculateAccuracy", () => {
  it("should display the percent correct predictions", () => {
    const neuralNetwork = () => [1, 0, 0];
    const inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const weights = [];
    const expectedPredictions = [
      [1, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [0, 0, 1],
      [1, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 0, 0]
    ];
    const expectedOutput = 0.8;

    assert.equal(
      calculateAccuracy(inputs, weights, expectedPredictions, neuralNetwork),
      expectedOutput
    );
  });
});

describe("findLargestIndex", () => {
  it("should find the index of the largest value of an array", () => {
    const array = [0, 1, 2, 9, 4];
    const expectedOutput = 3;

    assert.equal(findLargestIndex(array), expectedOutput);
  });
});
