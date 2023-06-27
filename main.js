var onnxSession;

startOnnxSession = function() {
	ort.InferenceSession.create("onnx/gpt2-lm-head-10.onnx", {executionProviders: ["wasm"] }).then((session) => {
		onnxSession = session;
		Module.onnx("status", "dummy");
	});
}

onnxInference = async function(textInput) {
	Module.onnx("inference", textInput);
	var str = textInput;
	var iterate = new Boolean(true);
	var n = 0;
	while (n < 256 && iterate) {
		n++;
		// setTimeout(async function(){
			var chars = '].!;?)"`';
			if (chars.indexOf(str.charAt(str.length - 1)) == 1) {
				Module.onnx("inference", str);
			}
			var tokens = GPTTokenizer_p50k_edit.encode(str);
			tokens = tokens.slice(Math.max(tokens.length - 50, 0))
			var conv = [];
			for (let i = 0; i < tokens.length; i++) {
				conv[i] = BigInt(tokens[i]);
			}
			const tensorA = new ort.Tensor("int64", conv, [1, 1, tokens.length]);
			const feeds = { input1: tensorA };
			const results = await onnxSession.run(feeds);
			var dataC = results["output1"].data;
			var dataA = dataC.slice(50257 * (tokens.length - 1), 50257 + 50257 * (tokens.length - 1));
			var entries = Object.entries(dataA);
			var sorted = entries.sort((a, b) => b[1] - a[1]);
			var newWord = GPTTokenizer_p50k_edit.decode([parseInt(sorted[randomWithProbability()][0])]);
			newWord = newWord.replace(/(\r\n|\n|\r)/gm, "");
			if (newWord != "<|endoftext|>") {
				str = str + newWord;
				console.log(n, str);
			} else {
				iterate = false;
			}
		// }, 0);
	}
}

function randomWithProbability() {
	var notRandomNumbers = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2];
	var idx = Math.floor(Math.random() * notRandomNumbers.length);
	return notRandomNumbers[idx];
}
