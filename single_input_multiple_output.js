const {
  scalarVectorMultiply,
  vectorSubtract,
  calculateErrors
} = require("./deep_learning_functions");

function neuralNetwork(input, weights) {
  return scalarVectorMultiply(input, weights);
}

let weights = [0.3, 0.2, 0.9];
const input = 0.65;
const expectedPrediction = [0.1, 1, 0.1];
const alpha = 0.1;
const errorThreshold = 0.0000000000000001;

while (true) {
  const prediction = neuralNetwork(input, weights);
  const predictionDelta = vectorSubtract(prediction, expectedPrediction);
  const errors = calculateErrors(prediction, expectedPrediction);
  const weightedPredictionDelta = scalarVectorMultiply(input, predictionDelta);
  const limitedWeightedPredictionDelta = scalarVectorMultiply(
    alpha,
    weightedPredictionDelta
  );

  weights = vectorSubtract(weights, limitedWeightedPredictionDelta);

  console.log("prediction", prediction);

  const stillLearning = errors.reduce((countStillLearning, error) => {
    if (error > errorThreshold) return countStillLearning + 1;
    return countStillLearning;
  }, 0);

  if (stillLearning === 0) break;
}
