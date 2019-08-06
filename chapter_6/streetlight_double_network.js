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

const { streetlightTesting, streetlightTraining } = generateStreetlightData(
  1000
);
const alpha = 0.01;
const hiddenSize = 4;
let layer1Weights = createRandomMatrix(
  hiddenSize,
  streetlightTesting.inputs[0].length
);
let layer2Weights = createRandomMatrix(1, hiddenSize);

for (let iteration = 0; iteration < 600; iteration = iteration + 1) {
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

    console.log(
      "prediction: ",
      Math.round(layer2),
      "output: ",
      streetlightTraining.outputs[streetlightsIndex]
    );
  }

  if (iteration % 10 === 0) {
    console.log("error: ", layer2Errors);
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
