const { createMatrix } = require("./deep_learning_functions");

const matrix = createMatrix(3, 5);
const firstRow = matrix[0];
firstRow[0] = 0;
console.log(firstRow);
