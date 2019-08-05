const {
  createRandomMatrix,
  dotVectorMatrix,
  relu,
  vectorSubtract,
  transpose,
  vectorMultiply,
  reluToDerivative,
  matrixSubtract,
  outerProduct
} = require("../deep_learning_functions");

const { streetlightTesting, streetlightTraining } = generateStreetlightData(4);
const hiddenLayerSize = 4;
let layer1Weights = createRandomMatrix(
  streetlightTesting.inputs[0].length,
  hiddenLayerSize
);
const iterationCount = 60;
let layer2Weights = createRandomMatrix(
  hiddenLayerSize,
  streetlightTraining.outputs[0].length
);
const alpha = 0.01;

for (
  let iterations = 0;
  iterations < iterationCount;
  iterations = iterations + 1
) {
  let totalErrors = 0;

  streetlightTraining.inputs.forEach((streetlights, streetlightsIndex) => {
    let hiddenLayerPredictions = dotVectorMatrix(streetlights, layer1Weights);

    hiddenLayerPredictions = relu(hiddenLayerPredictions);

    const finalPrediction = dotVectorMatrix(
      relu(hiddenLayerPredictions),
      layer2Weights
    );
    let finalDelta = vectorSubtract(
      finalPrediction,
      streetlightTraining.outputs[streetlightsIndex]
    );

    totalErrors = finalDelta.reduce(
      (total, delta) => total + Math.pow(delta, 2),
      totalErrors
    );

    let layer1Delta = dotVectorMatrix(finalDelta, transpose(layer2Weights));

    layer1Delta = vectorMultiply(
      layer1Delta,
      reluToDerivative(hiddenLayerPredictions)
    );
    finalDelta = outerProduct(finalDelta, hiddenLayerPredictions);
    layer1Delta = outerProduct(streetlights, layer1Delta);
    layer2Weights = matrixSubtract(layer2Weights, transpose(finalDelta));
    layer1Weights = matrixSubtract(layer1Weights, layer1Delta);

    console.log(
      finalPrediction,
      streetlightTraining.outputs[streetlightsIndex]
    );
  });
}

function generateStreetlightData(numberToCreate) {
  const streetlightTraining = {
    inputs: [],
    outputs: []
  };
  const streetlightTesting = {
    inputs: [],
    outputs: []
  };

  for (let _count = 0; _count < numberToCreate; _count = _count + 1) {
    const trainingLeft = Math.round(Math.random());
    const trainingCenter = Math.round(Math.random());
    const trainingRight = Math.round(Math.random());
    const testingLeft = Math.round(Math.random());
    const testingCenter = Math.round(Math.random());
    const testingRight = Math.round(Math.random());

    streetlightTraining.inputs.push([
      trainingLeft,
      trainingCenter,
      trainingRight
    ]);
    streetlightTraining.outputs.push([trainingCenter]);
    streetlightTesting.inputs.push([testingLeft, testingCenter, testingRight]);
    streetlightTesting.outputs.push([testingCenter]);
  }

  return {
    streetlightTesting,
    streetlightTraining
  };
}

function calculateAccuracy(testingData, layer1Weights, layer2Weights) {
  testingData.inputs.forEach((inputs, index) => {});
}

function neuralNetwork(inputs, weights) {
  return dotVectorMatrix(inputs, weights);
}
