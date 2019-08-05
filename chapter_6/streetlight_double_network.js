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

const { streetlightTesting, streetlightTraining } = generateStreetlightData(1);
const alpha = 0.01;
const hiddenSize = 4;
const layer1Weights = createRandomMatrix(
  hiddenSize,
  streetlightTesting.inputs[0].length
);
const layer2Weights = createRandomMatrix(1, hiddenSize);

for (let iteration = 0; iteration < 1; iteration = iteration + 1) {
  let layer2Errors = 0;

  for (
    let streetlightsIndex = 0;
    streetlightsIndex < streetlightTraining.inputs.length;
    streetlightsIndex = streetlightsIndex + 1
  ) {
    const layer0 = streetlightTraining.inputs[streetlightsIndex];
    const layer1 = relu(dotVectorMatrix(layer0, layer1Weights));
    const layer2 = dotVectorMatrix(layer1, layer2Weights);

    layer2Errors =
      layer2Errors +
      Math.pow(layer2[0] - streetlightTraining.outputs[streetlightsIndex], 2);
    console.log(layer2Errors);
  }
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
