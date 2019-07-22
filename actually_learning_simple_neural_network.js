function neuralNetwork(input, weights) {
  return scalarVectorMultiply(input, weights);
}

function vectorMultiply(vector1, vector2) {
  if (vector1.length !== vector2.length)
    throw new Error("vectors must be same length");

  return vector1.map((number, index) => number * vector2[index]);
}

function vectorSum(vector) {
  return vector.reduce((sum, number) => sum + number, 0);
}

function vectorSubtraction(vector1, vector2) {
  if (vector1.length !== vector2.length)
    throw new Error("vectors must be the same length");

  return vector1.map((number, index) => number - vector2[index]);
}

function dot(vector1, vector2) {
  return vectorSum(vectorMultiply(vector1, vector2));
}

function scalarVectorMultiply(scalar, vector) {
  return vector.map(number => number * scalar);
}

const input = 0.5;
let weights = [0.1, 0.9, 0.5];
const knownResults = [1, 1.2, 0.2];
const alpha = 0.01; // weights change limiter

// while (true) {
for (let count = 0; count < 5; count = count + 1) {
  const predictions = neuralNetwork(input, weights);
  const predictionDeltas = vectorSubtraction(predictions, knownResults);
  const errors = predictionDeltas.map(number => Math.pow(number, 2));
  const predictionDerivatives = scalarVectorMultiply(input, errors);

  console.log(
    predictions
    // predictionDelta,
    // error,
    // predictionDerivative,
    // weights
  );

  weights = weights.map(
    (currentWeight, index) =>
      currentWeight - predictionDerivatives[index] * alpha
  );

  if (Math.abs(predictionDeltas) < 0.000001) break;
}
