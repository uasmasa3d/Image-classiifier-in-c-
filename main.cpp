#include<iostream>
#include<unordered_map>
#include<string>
#include<vector>
#include <iterator>
#include"string.h"
#include"math.h"
#include "Source.cpp"
using namespace std;

int main() {
	// Testing print matrix
	vector<vector<double>> W;
	vector<double> w1;
	w1.push_back(5);
	w1.push_back(5);
	w1.push_back(5);
	w1.push_back(5);
	vector<double> w2;
	w2.push_back(5);
	w2.push_back(5);
	w2.push_back(5);
	w2.push_back(5);
	W.push_back(w1);
	W.push_back(w2);
	printMatrix(W);
	// Testing randomizeWeights
	
	W = randomizeWeights(W, 3,3);
	printMatrix(W);
	// Testing initialize_parameters_deep
	vector<int> layer_dims;
	layer_dims.push_back(4);
	layer_dims.push_back(8);
	layer_dims.push_back(8);
	layer_dims.push_back(5);
	layer_dims.push_back(1);
	unordered_map<string, vector<vector<double>>> parms = initialize_parameters_deep(layer_dims);
	printParams(parms);
	// Testing dot
	cout<<endl<< "matrix A: " << endl;
	vector<vector<double>> A;
	vector<double> w3;
	w3.push_back(5);
	w3.push_back(5);
	w3.push_back(5);
	w3.push_back(5);
	vector<double> w4;
	w4.push_back(5);
	w4.push_back(5);
	w4.push_back(5);
	w4.push_back(5);
	A.push_back(w3);
	A.push_back(w4);
	printMatrix(A);
	cout <<endl<< "matrix B: " << endl;
	vector<vector<double>>B;
	vector<double> w5;
	w5.push_back(2);
	w5.push_back(2);
	w5.push_back(3);
	w5.push_back(4);
	vector<double> w6;
	w6.push_back(9);
	w6.push_back(7);
	w6.push_back(6);
	w6.push_back(4);
	B.push_back(w5);
	B.push_back(w6);
	printMatrix(B);
	vector<vector<double>>C = dot(A, B);
	cout << endl << "matrix C: " << endl;
	printMatrix(C);
	//Test shape
	vector<int> shapeofC;
	shapeofC = shape(C);
	cout << "shape of C is: ( " << shapeofC[0] << " , " << shapeofC[1] << " )" << endl;
	// Test linear_forward_Z
	vector<vector<double>> b;
	vector<double> b1, b2;
	b1.push_back(3);
	b2.push_back(2);
	b.push_back(b1);
	b.push_back(b2);
	cout << endl << "matrix b: " << endl;
	printMatrix(b);
	vector<vector<double>> Z = linearForward_Z(A,B,b);
	cout << endl << "matrix Z: " << endl;
	printMatrix(Z);
	// Test linearForward_cache
	unordered_map<string, vector<vector<double>>> t_cache = linearForward_cache(A,B,b);
	printMatrix(t_cache["A"]);
	printMatrix(t_cache["W"]);
	printMatrix(t_cache["b"]);
	// Test sigmoid and relu
	double r = sigmoid(0),reluout = relu(-8);
	cout << "output of simoid = " << r << endl << "output of relu =" << reluout;
	// Test linear_activation_forward_A
	vector<vector<double>> laS=Z, laR = Z;
	cout << "linear activation sigmoid" << endl;
	laS = linear_activation_forward_A(A,B,b, 0);
	printMatrix(laS);
	cout << "linear activation relu" << endl;
	laR = linear_activation_forward_A(A, B, b, 1);
	printMatrix(laR);
	// Test linear_activation_forward_cache
	vector<unordered_map<string, vector<vector<double>>>> cache_laF = linear_activation_forward_cache(A, B, b, 0);
	unordered_map<string, vector<vector<double>>> linear_cache = cache_laF[0], activation_cache=cache_laF[1];
	cout << "linear cache: " << endl;
	printParams(linear_cache);
	cout << "activation_cache: " << endl;
	printParams(activation_cache);
	// Test L_model_forward_AL
	// cannot be tested without right shape
	// will work after editing the dot function to multiply non-square matrcies
	/*vector<vector<double>> X;
	randomizeWeights(X,4,8);
	vector<vector<double>> AL = L_model_forward_AL(X,parms);
	*/
	// Test L_model_forward_caches
	// also depends on dot
	

	// Test compute cost
	// from reference, foor Y = [[1,1,0]], AL = [[0.8],[0.9],[0.4]]. output should be 0.2797765635793422
	vector<vector<double>> Lastlayer, Y;
	vector<double> AL1, AL2, AL3, Y1;
	AL1.push_back(0.8);
	AL2.push_back(0.9);
	AL3.push_back(0.4);
	Lastlayer.push_back(AL1);
	Lastlayer.push_back(AL2);
	Lastlayer.push_back(AL3);
	Y1.push_back(1);
	Y1.push_back(1);
	Y1.push_back(0);
	Y.push_back(Y1);
	cout << "Y:";
	printMatrix(Y);
	cout << "AL:";
	printMatrix(Lastlayer);
	double cost = computeCost(Lastlayer, Y);
	cout << cost << endl;
	// Test Transpose
	vector<int> shapeBeforeTranspose = shape(Z), shapeAfterTranspose;
	Z = Transpose(Z);
	shapeAfterTranspose = shape(Z);
	printMatrix(Z);
	cout << "output of transpose is:" << endl;
	printMatrix(Z);
	cout << "Shape before transpose is: ( " << shapeBeforeTranspose[0] << " , " << shapeBeforeTranspose[1] << " )" << endl;
	cout << "Shape after transpose is: ( " << shapeAfterTranspose[0] << " , " << shapeAfterTranspose[1] << " )" << endl;
	
	// The rest of the code cannot run or be tested without the dot funtion.
	// Test linear_backward
	// Test cannot run without the modification required on dot
	/*vector<vector<double>> dZ, Ab, Wb,bb;
	dZ = randomizeWeights(dZ,3,4);
	Ab = randomizeWeights(Ab, 5, 4);
	Wb = randomizeWeights(Wb, 3, 5);
	bb = randomizeWeights(bb, 3, 1);
	unordered_map<string, vector<vector<double>>> linear_cacheb;
	linear_cacheb["W"] = Wb;
	linear_cacheb["b"] = bb;
	linear_cacheb["A"] = Ab;
	vector<vector<vector<double>>> t_output = linear_backward(dZ, linear_cacheb);
	vector<vector<double>> dA_prev = t_output[0], dW = t_output[1], db = t_output[2];
	cout << "dA_prev" << endl;
	printMatrix(dA_prev);
	cout << "dW" << endl;
	printMatrix(dW);
	cout << "db" << endl;
	printMatrix(db);
	*/
	// Test relu_backward and sigmoid_backward, I actually don't know how to test those


}