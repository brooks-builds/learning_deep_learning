const chai = require("chai");

const {
  vectorMultiply,
  scalarVectorMultiply,
  vectorSubtract,
  calculateErrors
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
