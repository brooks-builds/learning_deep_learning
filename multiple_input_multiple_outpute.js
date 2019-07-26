const {
  vectorMatrixMultiply,
  vectorSubtract,
  calculateErrors,
  outerProduct,
  scalarMatrixMultiply,
  matrixSubtract
} = require("./deep_learning_functions");

function neuralNetwork(inputs, weights) {
  return vectorMatrixMultiply(inputs, weights);
}

function stillLearning(errors, errorThreshold) {
  return (
    errors.reduce((countStillLearning, error) => {
      if (error > errorThreshold) return countStillLearning + 1;
      return countStillLearning;
    }, 0) > 0
  );
}

const inputs = [8.5, 0.65, 1.2];
const whatReallyHappened = [0.1, 1, 0.1];
let weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0], [0, 1.3, 0.1]];
const alpha = 0.01;
const errorThreshold = 0.0000000000000001;
const maxIterations = 1000;

for (let _count = 0; _count < maxIterations; _count = _count + 1) {
  const predictions = neuralNetwork(inputs, weights);
  const predictionDeltas = vectorSubtract(predictions, whatReallyHappened);
  const predictionErrors = calculateErrors(predictions, whatReallyHappened);
  const weightedPredictionDeltas = outerProduct(predictionDeltas, inputs);
  const scaledWeightedPredictionDeltas = scalarMatrixMultiply(
    alpha,
    weightedPredictionDeltas
  );

  weights = matrixSubtract(weights, scaledWeightedPredictionDeltas);

  console.log(predictions);

  if (!stillLearning(predictionErrors, errorThreshold)) break;
}
