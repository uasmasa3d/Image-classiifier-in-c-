//
//  main.cpp
//  ANNHelper
//
//  Created by Ziad Amerr on 14/12/2022.
//

#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <iterator>
#include "string.h"
#include "math.h"
#include "../Source.cpp"
using namespace std;

int main(int argc, const char * argv[]) {
    Source mySrc = Source();
    // Testing print matrix
    vector<double> w1 = {
        5,
        5,
        5,
        5
    };
    
    vector<double> w2 = {
        5,
        5,
        5,
        5
    };
    
    vector<vector<double>> W = {w1, w2};
    
    
    mySrc.printMatrix(W);
    // Testing randomizeWeights
    
    W = mySrc.randomizeWeights(W, 3,3);
    mySrc.printMatrix(W);
    // Testing initializeParametersDeep
    vector<int> layer_dims = {4, 8, 8, 5, 1};
    
    
    unordered_map<string, vector<vector<double>>> parms;
    parms = mySrc.initializeParametersDeep(layer_dims);
    mySrc.printParams(parms);
    
    
    // Testing dot
    cout<<endl<< "matrix A: " << endl;
    vector<double> w3 = {5, 5, 5, 5};
    vector<double> w4 = {5, 5, 5, 5};
    vector<vector<double>> A = {w3, w4};
    mySrc.printMatrix(A);
    
    
    
    cout << endl << "matrix B: " << endl;
    vector<double> w5 = {2, 2, 3, 4};
    vector<double> w6 = {9, 7, 6, 4};
    vector<vector<double>> B = {w5, w6};
    mySrc.printMatrix(B);
    
    
    
    cout << endl << "matrix C: " << endl;
    vector<vector<double>> C = mySrc.dot(A, B);
    mySrc.printMatrix(C);
    
    
    
    //Test shape
    vector<int> shapeofC;
    cout << "shape of C is: ( " << shapeofC[0] << " , " << shapeofC[1] << " )" << endl;
    shapeofC = mySrc.shape(C);
    
    
    // Test linear_forward_Z
    cout << endl << "matrix b: " << endl;
    vector<double> b1 = {3};
    vector<double> b2 = {2};
    vector<vector<double>> b = {b1, b2};
    mySrc.printMatrix(b);
    
    
    
    cout << endl << "matrix Z: " << endl;
    vector<vector<double>> Z = mySrc.linearForwardZ(A,B,b);
    mySrc.printMatrix(Z);
    // Test linearForward_cache
    unordered_map<string, vector<vector<double>>> t_cache;
    t_cache = mySrc.linearForwardCache(A,B,b);
    mySrc.printMatrix(t_cache["A"]);
    mySrc.printMatrix(t_cache["W"]);
    mySrc.printMatrix(t_cache["b"]);
    // Test sigmoid and relu
    double r = mySrc.sigmoid(0);
    double reluout = mySrc.relu(-8);
    cout << "output of simoid = " << r << endl << "output of relu =" << reluout;
    // Test linear_activation_forward_A
    vector<vector<double>> laS=Z, laR = Z;
    cout << "linear activation sigmoid" << endl;
    laS = mySrc.linearActivationForwardA(A,B,b, 0);
    mySrc.printMatrix(laS);
    cout << "linear activation relu" << endl;
    laR = mySrc.linearActivationForwardA(A, B, b, 1);
    mySrc.printMatrix(laR);
    // Test linear_activation_forward_cache
    vector<unordered_map<string, vector<vector<double>>>> cache_laF = mySrc.linearActivationForwardCache(A, B, b, 0);
    unordered_map<string, vector<vector<double>>> linear_cache = cache_laF[0], activation_cache=cache_laF[1];
    cout << "linear cache: " << endl;
    mySrc.printParams(linear_cache);
    cout << "activation_cache: " << endl;
    mySrc.printParams(activation_cache);
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
    mySrc.printMatrix(Y);
    cout << "AL:";
    mySrc.printMatrix(Lastlayer);
    double cost = mySrc.computeCost(Lastlayer, Y);
    cout << cost << endl;
    // Test Transpose
    vector<int> shapeBeforeTranspose = mySrc.shape(Z), shapeAfterTranspose;
    Z = mySrc.transpose(Z);
    shapeAfterTranspose = mySrc.shape(Z);
    mySrc.printMatrix(Z);
    cout << "output of transpose is:" << endl;
    mySrc.printMatrix(Z);
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

