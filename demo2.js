let recognizer;

function predictWord() {
 // Array of words that the recognizer is trained to recognize.
 const words = recognizer.wordLabels();
 recognizer.listen(({scores}) => {
   // Turn scores into a list of (score,word) pairs.
   scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // Find the most probable word.
   scores.sort((s1, s2) => s2.score - s1.score);
   document.querySelector('#console').textContent = scores[0].word;
 }, {probabilityThreshold: thresholdPer/100});
}

async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 buildModel();
 draw( null );
}

// One frame is ~23ms of audio.
const NUM_FRAMES      =   3;
const FFT_RESULT_LEN  = 232;
let examples          =  [];
let epochLimit        =  10;
var dbThreshold       = -40;
var thresholdPer      =  80; // %

function collect(label) {
 if (recognizer.isListening()) {
   document.getElementById("button0").style.color = "green";
   document.getElementById("button1").style.color = "green";
   document.getElementById("button2").style.color = "green";
   return recognizer.stopListening();
 }
 if ( label == null ) {
   return;
 }

 if ( label == 0 ) {
   document.getElementById("button0").style.color = "red";
 } else if ( label == 1 ) {
   document.getElementById("button1").style.color = "red";
 } else if ( label == 2 ) {
   document.getElementById("button2").style.color = "red";
 }

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   var addValue     = false;
   var index        = 0;
   var dataSubArray = null;

   if (label != 2) {
     var result = peakDbThresholdAndPositionCheck( data, dbThreshold  );
     index         = result['index'];
     var threshold = result['threshold'];
     if ( threshold == true && index > 0  && index < 41 ) {
       var start = (index-1)*FFT_RESULT_LEN;
       var end   = FFT_RESULT_LEN*(index+2);
       dataSubArray = data.subarray( start, end );
       addValue =true;
     }
   } else {
     peakDb(data);
     dataSubArray = data.subarray(-frameSize * NUM_FRAMES);
     addValue =true;
   }

   //console.log("Add value ", addValue, index, threshold );
   if ( addValue == false || dataSubArray == null ) {
     //console.log("No addition");
     return;
   }
   let vals = normalize(dataSubArray);
   draw( dataSubArray );

   examples.push({vals, label});
   document.getElementById('captureCountValue').textContent = examples.length;
   //document.querySelector('#console').textContent =
   //    `${examples.length} Training Data Samples collected`;
 }, {
   overlapFactor: 0.1, ///0.999, //0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

function captureReset() {
  examples = [];
  document.getElementById('captureCountValue').textContent = examples.length;
}

function myFunction(item) {
  document.getElementById("demo").innerHTML = numbers.reduce(getSum, 0);
}

function thresholdCheck(data, threshold) {
  function getSum(total, num) {
    return total + num;
  }
  var dataAbs = data.map(function(x) { return Math.abs(x); });
  var sum = data.reduce( getSum, 0);
  if (sum > threshold ) { return false; }
  else { return true; }
}

function peakDbThresholdCheck(data, threshold) {
  function peakCheck(peak, num) {
    if ( num > peak ) {
      peak = num;
    }
    return peak;
  }

  var peak = data.reduce( peakCheck, -200);
  if (peak > threshold && peak < 0) {
    //console.log(peak);
    return true;
  }
  else { return false; }
}

function peakDbThresholdAndPositionCheck(data, threshold) {
  function peakCheck(peak, num) {
    if ( num > peak ) {
      peak = num;
    }
    return peak;
  }

  var mainPeakIndex = 0;
  var peak = -200;

  const startOffset = 20;
  const endOffset   = 20;


  for( var index = 0; index < 43; index++) {
    var start = index*FFT_RESULT_LEN+startOffset;
    var end   = FFT_RESULT_LEN*(index+1) -(1 + endOffset);
    dataSubArray = data.subarray( start, end );
    localPeak = dataSubArray.reduce( peakCheck, -200);
    // console.log( start, end );
    // console.log(index, localPeak, peak, mainPeakIndex );

    if ( localPeak >= peak ) {
      peak          = localPeak;
      mainPeakIndex = index;
    }
  }
  // Display peak
  document.getElementById('currentVolume').textContent = peak.toFixed(1);

  if ( peak > threshold && peak < 0) {
    //console.log(peak, mainPeakIndex);
    return { "threshold": true, "index": mainPeakIndex };
  }
  else {
    return { "threshold": false, "index": 0 };
  }
}

function peakDb(data) {
  function peakCheck(peak, num) {
    if ( num > peak ) {
      peak = num;
    }
    return peak;
  }

  var mainPeakIndex = 0;
  var peak = -200;

  const startOffset = 20;
  const endOffset   = 20;


  for( var index = 0; index < 43; index++) {
    var start = index*FFT_RESULT_LEN+startOffset;
    var end   = FFT_RESULT_LEN*(index+1) -(1 + endOffset);
    dataSubArray = data.subarray( start, end );
    localPeak = dataSubArray.reduce( peakCheck, -200);
    // console.log( start, end );
    // console.log(index, localPeak, peak, mainPeakIndex );

    if ( localPeak >= peak ) {
      peak          = localPeak;
      mainPeakIndex = index;
    }
  }
  // Display peak
  document.getElementById('currentVolume').textContent = peak.toFixed(1);
}

function thresholdPerChange() {
  thresholdPer = document.getElementById('thresholdPer').value;
}

function soundThresholdCheck(data, threshold) {
  function soundAverage(peak, num) {
    var absNum = Math.abs(num);
    if ( absNum > peak ) {
      peak = absNum;
    }
    return peak;
  }

  var peak = data.reduce( peakCheck, 0);
  //console.log(peak);
  if (peak > threshold ) { return true; }
  else { return false; }
}


function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}

const INPUT_SHAPE = [NUM_FRAMES, FFT_RESULT_LEN, 1];
let model;

async function train() {
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 3);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

 await model.fit(xs, ys, {
   batchSize: 5,
   epochs: epochLimit,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.getElementById('epochs').textContent = 'Epoch:'+ (epoch + 1) + ' of ' + epochLimit;
       document.getElementById('accuracy').textContent = 'Accuracy:'+ ( logs.acc * 100).toFixed(1);
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}

function thresholdChange() {
   dbThreshold = document.getElementById('thresholdDb').value;
}

function epochChange() {
   epochLimit = document.getElementById('epochLimit').value;
}

function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));

 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

function reset() {
  document.getElementById('output').value = 5;
}

async function moveSlider(labelTensor) {
 const label = (await labelTensor.data())[0];
 //document.getElementById('console').textContent = label;
 if (label == 2) {
   return;
 }

 let delta = 0.1;
 const prevValue = +document.getElementById('output').value;
 document.getElementById('output').value =
     prevValue + (label === 0 ? -delta : delta);
}

async function displayLabel(labelTensor) {
 const label = (await labelTensor.data())[0];

 document.getElementById("dispButton0").style.color = "green";
 document.getElementById("dispButton1").style.color = "green";
 document.getElementById("dispButton2").style.color = "green";

 if ( label == 0 ) {
   document.getElementById("dispButton0").style.color = "red";
 } else if ( label == 1 ) {
   document.getElementById("dispButton1").style.color = "red";
 } else if ( label == 2 ) {
   document.getElementById("dispButton2").style.color = "red";
 }

 //if ( label == 2 ) {
 //   document.getElementById('console').textContent = " ";
 //} else {
 //   document.getElementById('console').textContent = label;
 //}
}

function listen() {
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   document.getElementById('listen').textContent = 'Listen';
   return;
 }
 toggleButtons(false);
 document.getElementById('listen').textContent = 'Stop';
 document.getElementById('listen').disabled = false;

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   draw( data );
   const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
   const probs = model.predict(input);
   const predLabel = probs.argMax(1);
   await moveSlider(predLabel);
   await displayLabel(predLabel);
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.98, // 0.999, //999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

// FFT or Raw Audio Data Display
function draw( dataArray ) {

    var canvas    = document.getElementById("myCanvas");
    var canvasCtx = canvas.getContext("2d");

    canvasCtx.fillStyle = 'rgb(255, 255, 255)';
    canvasCtx.fillRect(0, 0, 696, 200);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

    // Box width
    var bw = FFT_RESULT_LEN * NUM_FRAMES;
    if ( dataArray != null) {
      bw = dataArray.length;
    }
    // Box height
    var bh = 200;
    // Padding
    var p = 0;

    // Draw Grid
    canvasCtx.beginPath();
    canvasCtx.lineWidth = 0.2;              // Thin line
    for (var x = 0; x <= bw; x += 40) {
      canvasCtx.moveTo(0.5 + x + p, p);
      canvasCtx.lineTo(0.5 + x + p, bh + p);
    }
    for (var x = 0; x <= bh; x += FFT_RESULT_LEN/10 ) {
      canvasCtx.moveTo(p, 0.5 + x + p);
      canvasCtx.lineTo(bw + p, 0.5 + x + p);
    }
    canvasCtx.strokeStyle='#888888';
    canvasCtx.stroke(); // Draw

    // FFT Division
    canvasCtx.beginPath();
    canvasCtx.lineWidth = 0.4;              // Thin line
    for (var x = 0; x <= bw; x += FFT_RESULT_LEN) {
      canvasCtx.moveTo(0.5 + x + p, p);
      canvasCtx.lineTo(0.5 + x + p, bh + p);
    }
    canvasCtx.strokeStyle='#444444';
    canvasCtx.stroke(); // Draw

    if ( dataArray == null) {
      return;
    }
    // Draw audio data
    canvasCtx.beginPath();
    canvasCtx.lineWidth = 1;
    var start      = 0;
    var end        = dataArray.length;
    //console.log( dataLength );
    for(var i = start; i < end; i++) {
      var y = 0
      y = dataArray[i];
      y = -y;

      if(i === 0) {
        canvasCtx.moveTo(i, y);
      } else {
        canvasCtx.lineTo(i, y);
      }
    }
    canvasCtx.strokeStyle='#000000';
    canvasCtx.stroke(); // Draw

    // text
    canvasCtx.font = "10px Arial";
    canvasCtx.strokeText("FFT 1",FFT_RESULT_LEN*0 + 15, 15);
    canvasCtx.strokeText("FFT 2",FFT_RESULT_LEN*1 + 15, 15);
    canvasCtx.strokeText("FFT 3",FFT_RESULT_LEN*2 + 15, 15);
    canvasCtx.stroke(); // Draw

  };


app();
