var onnxSession;
var calls = 0;

startOnnxSession = function() {
	ort.InferenceSession.create("onnx/gpt2-lm-head-10.onnx", {executionProviders: ["wasm"] }).then((session) => {
		onnxSession = session;
		Module.onnx("status", "dummy");
	});
}

onnxInference = function(textInput) {
	gptLoop(textInput, ++calls);
}

async function gptLoop(textInput, id) {
	Module.onnx("inference", textInput);
	var textOutput = textInput;
	var generatedToken = 0;
	var bigInt64Array = [];
	var tokens = GPTTokenizer_p50k_edit.encode(textOutput);
	for (let i = 0; i < tokens.length; i++) {
		bigInt64Array[i] = BigInt(tokens[i]);
	}
	while (generatedToken < 256) {
		await delay(0);
		if (id !== calls) break;
		generatedToken++;
		bigInt64Array = bigInt64Array.slice(Math.max(bigInt64Array.length - 50, 0))
		const tensorA = new ort.Tensor("int64", bigInt64Array, [1, 1, bigInt64Array.length]);
		const feeds = { input1: tensorA };
		const results = await onnxSession.run(feeds);
		var dataC = results["output1"].data;
		var dataA = dataC.slice(50257 * (bigInt64Array.length - 1), 50257 + 50257 * (bigInt64Array.length - 1));
		var entries = Object.entries(dataA);
		var sorted = entries.sort((a, b) => b[1] - a[1]);
		var newToken = parseInt(sorted[randomWithProbability()][0]);
		bigInt64Array.push(BigInt(newToken));
		var newWord = GPTTokenizer_p50k_edit.decode([newToken]);
		newWord = newWord.replace(/(\r\n|\n|\r)/gm, " ");
		if (newWord != "<|endoftext|>") {
			textOutput = textOutput + newWord;
			var chars = '].!;?)`';
			var lastChar = textOutput.charAt(textOutput.length - 1);
			if (chars.indexOf(lastChar) == 1 || lastChar == "\"") {
				Module.onnx("inference", textOutput);
			}
			console.log("Loop number:", calls, "Token number:", generatedToken, "Generated text:", textOutput);
		} else {
			console.log("End of text!");
			break;
		}
	}
}

function delay(time) {
	return new Promise(resolve => {setTimeout(resolve, time)});
}

function randomWithProbability() {
	var notRandomNumbers = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3];
	var idx = Math.floor(Math.random() * notRandomNumbers.length);
	return notRandomNumbers[idx];
}
