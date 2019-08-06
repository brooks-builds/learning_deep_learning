const {
  createRandomMatrix,
  dotVectorMatrix,
  relu,
  vectorSubtract,
  transpose,
  vectorMultiply,
  reluToDerivative,
  matrixSubtract,
  outerProduct,
  matrixMultiply,
  dotMatrix,
  scalarMatrixMultiply
} = require("../deep_learning_functions");

const { streetlightTesting, streetlightTraining } = generateStreetlightData(5);
const alpha = 0.01;
const hiddenSize = 4;
let layer1Weights = createRandomMatrix(
  hiddenSize,
  streetlightTesting.inputs[0].length
);
let layer2Weights = createRandomMatrix(1, hiddenSize);

for (let iteration = 0; iteration < 1000; iteration = iteration + 1) {
  let layer2Errors = 0;

  for (
    let streetlightsIndex = 0;
    streetlightsIndex < streetlightTraining.inputs.length;
    streetlightsIndex = streetlightsIndex + 1
  ) {
    const layer0 = streetlightTraining.inputs[streetlightsIndex];
    const { layer2, layer1 } = neuralNetwork(
      layer0,
      layer1Weights,
      layer2Weights
    );

    layer2Errors =
      layer2Errors +
      Math.pow(layer2[0] - streetlightTraining.outputs[streetlightsIndex], 2);

    const layer2Delta =
      layer2[0] - streetlightTraining.outputs[streetlightsIndex];
    const layer1Delta = matrixMultiply(
      [dotVectorMatrix([layer2Delta], transpose(layer2Weights))],
      [reluToDerivative(layer1)]
    );

    layer2Weights = matrixSubtract(
      layer2Weights,
      scalarMatrixMultiply(
        alpha,
        dotMatrix(transpose([layer1]), [[layer2Delta]])
      )
    );

    layer1Weights = matrixSubtract(
      layer1Weights,
      scalarMatrixMultiply(alpha, dotMatrix(transpose([layer0]), layer1Delta))
    );
  }

  if (iteration % 10 === 0) {
    console.log("error: ", layer2Errors);
    console.log(
      "accuracy: ",
      checkAccuracy(streetlightTesting, layer1Weights, layer2Weights)
    );
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

function neuralNetwork(layer0, layer1Weights, layer2Weights) {
  const layer1 = relu(dotVectorMatrix(layer0, layer1Weights));
  const layer2 = dotVectorMatrix(layer1, layer2Weights);

  return {
    layer1,
    layer2
  };
}

function checkAccuracy(testingData, layer1Weights, layer2Weights) {
  let correctCount = 0;

  testingData.inputs.forEach((streetlights, index) => {
    const { layer2 } = neuralNetwork(
      streetlights,
      layer1Weights,
      layer2Weights
    );

    const prediction = Math.round(layer2[0]);
    const correctOutput = testingData.outputs[index][0];

    if (prediction === correctOutput) {
      correctCount = correctCount + 1;
    }
  });

  return (correctCount / testingData.outputs.length) * 100;
}
