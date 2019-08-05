const math = require("mathjs");

function relu(x) {
  return x > 0 ? x : 0;
}

function relu2deriv(output) {
  return output > 0 ? 1 : 0;
}

const streetlights = math.matrix([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]]);

const walkVsStop = math.transpose(math.matrix([[0, 1, 0, 1]]));

const alpha = 0.01;
const hiddenSize = 4;

let layer1Weights = math.zeros(3, hiddenSize);
layer1Weights = layer1Weights.map(() => math.random(-1, 1));

const layer2Weights = math.zeros(hiddenSize, 1).map(() => math.random(-1, 1));

for (let iteration = 0; iteration < 1; iteration = iteration + 1) {
  let totalErrors = 0;
  const [streetlightRows] = streetlights.size();

  for (
    let streetlightRow = 0;
    streetlightRow < streetlightRows;
    streetlightRow = streetlightRow + 1
  ) {
    const layer0 = math.row(streetlights, streetlightRow);
    const layer1 = [];
  }
}
