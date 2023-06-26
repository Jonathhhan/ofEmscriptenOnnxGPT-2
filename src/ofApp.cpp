#include "ofApp.h"

std::string onnxModelStatus;
std::string inferenceResult;
bool onnxModelIsLoaded;

void onnx(std::string info, std::string inference) {
	if (info == "status") {  
		onnxModelStatus = "Onnx model loaded. Type something and press enter.";
		onnxModelIsLoaded = true;
		
	} else if (info == "inference") {		
		inferenceResult = inference;
	}
}

//--------------------------------------------------------------
EMSCRIPTEN_BINDINGS(Module){
	emscripten::function("onnx", &onnx);
}

//--------------------------------------------------------------
void ofApp::setup(){
	textInput = "The meaning of life";
	paragraph2.setText("Start sequence: " + textInput);
	paragraph.setWidth(700);
	paragraph.setPosition(50, 150);
	paragraph.setColor(ofColor(250, 250, 250));
	paragraph.setBorderPadding(30);
	paragraph.setAlignment(ofxParagraph::ALIGN_LEFT);
	paragraph.setFont("data/fonts/mono.ttf", 12);
	paragraph2.setWidth(700);
	paragraph2.setPosition(50, 50);
	paragraph2.setColor(ofColor(250, 250, 250));
	paragraph2.setBorderPadding(30);
	paragraph2.setAlignment(ofxParagraph::ALIGN_LEFT);
	paragraph2.setFont("data/fonts/mono.ttf", 12);
	paragraph3.setWidth(700);
	paragraph3.setPosition(50, 750);
	paragraph3.setColor(ofColor(250, 250, 250));
	paragraph3.setBorderPadding(30);
	paragraph3.setAlignment(ofxParagraph::ALIGN_LEFT);
	paragraph3.setFont("data/fonts/mono.ttf", 12);
	onnxModelIsLoaded = false;
	ofSetBackgroundColor(100, 100, 100);
	onnxModelStatus = "Onnx model loading.";
	EM_ASM(
		startOnnxSession();
	);
}

//--------------------------------------------------------------
void ofApp::update(){
	paragraph.setText(inferenceResult);
	paragraph3.setText(onnxModelStatus);
}

//--------------------------------------------------------------
void ofApp::draw(){
	paragraph.draw();
	paragraph2.draw();
	paragraph3.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if (key == OF_KEY_RETURN) {
		if (onnxModelIsLoaded) {
			EM_ASM(onnxInference(UTF8ToString($0)), textInput.c_str());
		}
	} else {
		if (key == 8 && textInput.size() > 0) {
			textInput = textInput.substr(0, textInput.size() - 1);
		} else if (key == 127) {
			textInput = "";
		} else if (textInput.size() < 150) {
			textInput.append (1, (char)key );
		}
	}
	paragraph2.setText("Start sequence: " + textInput);
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
