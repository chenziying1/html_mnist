/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs-node');
const https = require('https');
const fs = require('fs');
const util = require('util');
const zlib = require('zlib');
const assert = require('assert');
const argparse = require('argparse');

// Helper function to read files as a promise
const readFile = util.promisify(fs.readFile);

// MNIST data constants:
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte';
const TEST_IMAGES_FILE = 't10k-images-idx3-ubyte';
const TEST_LABELS_FILE = 't10k-labels-idx1-ubyte';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

// Function to download and unzip files
async function fetchOnceAndSaveToDiskWithBuffer(filename) {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}.gz`;
    if (fs.existsSync(filename)) {
      resolve(readFile(filename));
      return;
    }
    const file = fs.createWriteStream(filename);
    console.log(`  * Downloading from: ${url}`);
    https.get(url, (response) => {
      const unzip = zlib.createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on('end', () => {
        resolve(readFile(filename));
      });
    });
  });
}

// Load header values for images and labels
function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

// Load MNIST images
async function loadImages(filename) {
  const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);

  const headerValues = loadHeaderValues(buffer, IMAGE_HEADER_BYTES);
  assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
  assert.equal(headerValues[2], IMAGE_HEIGHT);
  assert.equal(headerValues[3], IMAGE_WIDTH);

  const images = [];
  let index = IMAGE_HEADER_BYTES;
  while (index < buffer.byteLength) {
    const array = new Float32Array(IMAGE_FLAT_SIZE);
    for (let i = 0; i < IMAGE_FLAT_SIZE; i++) {
      array[i] = buffer.readUInt8(index++) / 255;
    }
    images.push(array);
  }

  assert.equal(images.length, headerValues[1]);
  return images;
}

// Load MNIST labels
async function loadLabels(filename) {
  const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);

  const headerValues = loadHeaderValues(buffer, LABEL_HEADER_BYTES);
  assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

  const labels = [];
  let index = LABEL_HEADER_BYTES;
  while (index < buffer.byteLength) {
    const array = new Int32Array(LABEL_RECORD_BYTE);
    array[0] = buffer.readUInt8(index++);
    labels.push(array);
  }

  assert.equal(labels.length, headerValues[1]);
  return labels;
}

// Define the MNIST dataset class
class MnistDataset {
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
  }

  async loadData() {
    this.dataset = await Promise.all([
      loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE),
      loadImages(TEST_IMAGES_FILE), loadLabels(TEST_LABELS_FILE)
    ]);
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
  }

  getTrainData() {
    return this.getData_(true);
  }

  getTestData() {
    return this.getData_(false);
  }

  getData_(isTrainingData) {
    let imagesIndex = isTrainingData ? 0 : 2;
    let labelsIndex = isTrainingData ? 1 : 3;

    const size = this.dataset[imagesIndex].length;
    const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

    let imageOffset = 0;
    let labelOffset = 0;
    for (let i = 0; i < size; ++i) {
      images.set(this.dataset[imagesIndex][i], imageOffset);
      labels.set(this.dataset[labelsIndex][i], labelOffset);
      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }
}

// Define the model
const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
model.add(tf.layers.flatten());
model.add(tf.layers.dropout({rate: 0.25}));
model.add(tf.layers.dense({units: 512, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.5}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

const optimizer = 'rmsprop';
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

// Training and evaluation function
async function run(epochs, batchSize, modelSavePath) {
  const dataset = new MnistDataset();
  await dataset.loadData();

  const {images: trainImages, labels: trainLabels} = dataset.getTrainData();
  model.summary();

  const numTrainExamplesPerEpoch = trainImages.shape[0] * (1 - 0.15);
  const numTrainBatchesPerEpoch = Math.ceil(numTrainExamplesPerEpoch / batchSize);

  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit: 0.15
  });

  const {images: testImages, labels: testLabels} = dataset.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(`\nEvaluation result:\nLoss = ${evalOutput[0].dataSync()[0].toFixed(3)}; Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

// Parse command-line arguments
const parser = new argparse.ArgumentParser({
  description: 'TensorFlow.js-Node MNIST Example.',
  add_help: true
});
parser.add_argument('--epochs', {
  type: 'int',
  default: 20,
  help: 'Number of epochs to train the model for.'
});
parser.add_argument('--batch_size', {
  type: 'int',
  default: 128,
  help: 'Batch size to be used during model training.'
});
parser.add_argument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});
const args = parser.parse_args();

// Run the app
run(args.epochs, args.batch_size, args.model_save_path);
