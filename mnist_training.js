const mnist = require("mnist");

const {
  createZerosMatrix,
  dot,
  calculateErrors,
  vectorSubtract,
  outerProduct,
  scalarMatrixMultiply,
  matrixSubtract
} = require("./deep_learning_functions");

const imageSet = mnist.set(8000, 2000);

const trainingImages = imageSet.training;
const testImages = imageSet.test;

// console.log(trainingImages[0].input, trainingImages[0].output);
// console.log(testImages[0].input, testImages[0].output);

let weights = createZerosMatrix(
  trainingImages[0].input.length,
  trainingImages[0].output.length
);
const alpha = 0.01;

for (
  let trainingImagesIndex = 0;
  trainingImagesIndex < 3;
  trainingImagesIndex = trainingImagesIndex + 1
) {
  const currentTrainingImage = trainingImages[trainingImagesIndex].input;
  const currentTrainingImageCorrectOutput =
    trainingImages[trainingImagesIndex].output;

  // calculate prediction
  const predictions = neuralNetwork(currentTrainingImage, weights);

  // calculate errors
  const errors = calculateErrors(
    predictions,
    currentTrainingImageCorrectOutput
  ); // [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

  // calculate deltas
  const deltas = vectorSubtract(predictions, currentTrainingImageCorrectOutput); // [0, 0, -1, 0, 0, 0, 0, 0, 0, 0]

  // calculate weighted deltas
  const weightedDeltas = outerProduct(currentTrainingImage, deltas);
  console.log(weightedDeltas); // first 10 in each row are fine, NaN after

  // update weights
  const limitedWeightedDeltas = scalarMatrixMultiply(alpha, weightedDeltas);
  const newWeights = matrixSubtract(weights, limitedWeightedDeltas);

  weights = newWeights;
  // has errors plateaued
  // how good are we with the testImages?
  // Should we keep going?
  //   console.log(predictions, currentTrainingImageCorrectOutput);
}

function neuralNetwork(inputs, weights) {
  return weights.map(weight => dot(inputs, weight));
}
