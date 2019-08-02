const {
  dot,
  scalarVectorMultiply,
  vectorSubtract
} = require("../deep_learning_functions");

const streetlight_configurations = [
  [1, 0, 1],
  [0, 1, 1],
  [0, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
  [1, 0, 1]
];
const did_walk = [[0], [1], [0], [1], [1], [0]];
let weights = [Math.random(), Math.random(), Math.random()];
const alpha = 0.1;

for (let _count = 0; _count < 40; _count = _count + 1) {
  let totalErrors = 0;
  streetlight_configurations.forEach((currentInput, index) => {
    const currentExpectedOutput = did_walk[index];
    const prediction = dot(currentInput, weights);
    const error = Math.pow(prediction - currentExpectedOutput[0], 2);
    const delta = prediction - currentExpectedOutput[0];
    const weightedDelta = scalarVectorMultiply(delta, currentInput);
    const limitedWeightedDelta = scalarVectorMultiply(alpha, weightedDelta);

    weights = vectorSubtract(weights, limitedWeightedDelta);
    totalErrors = totalErrors + error;
    console.log("prediction: ", prediction);
    console.log("expected output: ", currentExpectedOutput[0]);
  });
  console.log("total errors: ", totalErrors);
}
