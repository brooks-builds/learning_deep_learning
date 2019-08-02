const mnist = require("mnist");

const {
  createZerosMatrix,
  dot,
  calculateErrors,
  vectorSubtract,
  outerProduct,
  scalarMatrixMultiply,
  matrixSubtract,
  calculateAccuracy
} = require("./deep_learning_functions");

const imageSet = mnist.set(8000, 2000);

const trainingImages = imageSet.training;
const testImages = imageSet.test;
const testImagesInputs = testImages.map(imageObject => imageObject.input);
const testImagesOutputs = testImages.map(imageObject => imageObject.output);

// console.log(trainingImages[0].input, trainingImages[0].output);
// console.log(testImages[0].input, testImages[0].output);

let weights = createZerosMatrix(
  trainingImages[0].input.length,
  trainingImages[0].output.length
);
const alpha = 0.01;

function train() {
  let errors;

  for (
    let trainingImagesIndex = 0;
    trainingImagesIndex < trainingImages.length;
    trainingImagesIndex = trainingImagesIndex + 1
  ) {
    // const currentTrainingImage = trainingImages[trainingImagesIndex].input;
    const currentTrainingImage = trainingImages[0].input;
    // const currentTrainingImageCorrectOutput =
    // trainingImages[trainingImagesIndex].output;
    const currentTrainingImageCorrectOutput = trainingImages[0].output;

    // calculate prediction
    const predictions = neuralNetwork(currentTrainingImage, weights);

    // calculate errors
    errors = calculateErrors(predictions, currentTrainingImageCorrectOutput); // [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    // calculate deltas
    const deltas = vectorSubtract(
      predictions,
      currentTrainingImageCorrectOutput
    ); // [0, 0, -1, 0, 0, 0, 0, 0, 0, 0]

    // calculate weighted deltas
    const weightedDeltas = outerProduct(currentTrainingImage, deltas);

    // update weights
    const limitedWeightedDeltas = scalarMatrixMultiply(alpha, weightedDeltas);
    const newWeights = matrixSubtract(weights, limitedWeightedDeltas);

    weights = newWeights;
    // has errors plateaued
    // how good are we with the testImages?
    // Should we keep going?
  }

  return weights;
}

for (let count = 0; count < 10000; count = count + 1) {
  const newWeights = train();
  console.log(`training ${count} pass complete!`);
  const accuracy = calculateAccuracy(
    testImagesInputs,
    newWeights,
    testImagesOutputs,
    neuralNetwork
  );
  console.log("accuracy: ", accuracy);
  console.log("*****");
}

function neuralNetwork(inputs, weights) {
  return weights.map(weight => dot(inputs, weight));
}
