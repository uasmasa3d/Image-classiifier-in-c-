//
//  Source.cpp
//  ANNHelper
//
//  Created by Ziad Amerr on 14/12/2022.
//

// note that all images are converted into 1D array before everything starts, this contradicts some comments below but this is the right one so ignore any required modifications regarding this point.
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <iterator>
#include "math.h"

using namespace std;

/*
 Please use consistent naming scheme, the default adapted shceme by C++
 developers is camelCase and not snake_case, I will reedit all functions
 and provide pull request the changes.
 */



/*
 What is dim1, dim2? Why do I even need them? You have a W matrix,
 you can get the dim1 from W.size() and dim2 from W.at(0).size()
 if dim1 is the number of neurons of the first layer and dim2
 is the number of neurons in the second layer
 
 What is the output of randomizeWeights?
 We just need an initializer that initializer a draws a random weight
 float from random distribution

 We need
 double getRandomWeight()
 instead of
 v<v<double>> randomizeWeights(v<v<double>> &W, int dim1, int dim2)
 
 automation of weight initialization will be called from the NeuralNetwork class itself
*/
class Source {
public:
    vector<vector<double>>& randomizeWeights(vector<vector<double>> & W, int dim1, int dim2) {
        for (int j = 0; j < dim1; j++) {
            vector<double> S;
            for (int k = 0; k < dim2; k++) {
                S.push_back((rand()) * 0.1);
            }
            W.push_back(S);
        }
        return W;
    }
    
    /*
     Does this work?
     */
    void printMatrix(vector<vector<double>>& W) {
        for (int i = 0; i < W.size(); i++) {
            for (int j = 0; j < W[i].size(); j++)
                cout << W[i][j] << " ";
            cout << endl;
        }
    }
    
    /*
     map(str -> 2d_vector<params>) ?????
     We can simply loop over all neurons' weights
     */
    void printParams(unordered_map<string, vector<vector<double>>>& parameters) {
        unordered_map<string, vector<vector<double>>>::iterator itr;
        for (itr = parameters.begin(); itr != parameters.end(); ++itr) {
            cout << itr->first << endl;
            for (int i = 0; i < itr->second.size(); i++) {
                for (int j = 0; j < itr->second[i].size(); j++)
                    cout << itr->second[i][j] << " ";
                cout << endl;
            }
            
        }
        
    }
    
    /*
     Is this used to get the parameters or to initialize them with dummy variables?
     If it is used to initialize them, why is it returning an object and not void?
     And if it is initializing them, why are we using srand(5)?
     */
    unordered_map<string, vector<vector<double>>> initializeParametersDeep(vector<int> layerDims) {
        /*
         * Arguments:
         layer_dims --  array (list) containing the dimensions of each layer in our network
         
         Returns:
         parameters --  dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
         Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
         bl -- bias vector of shape (layer_dims[l], 1)
         */
        srand(5);
        unordered_map<string, vector<vector<double>>> parameters;
        for (int i = 1; i < layerDims.size(); i++) {
            vector<vector<double>> W,b;
            W = randomizeWeights(W, layerDims[i], layerDims[i - 1]);
            parameters.insert(make_pair("W" + to_string(i), W));
            b = randomizeWeights(b, layerDims[i], 1);
            parameters.insert(make_pair("b" + to_string(i), b));
        }
        return parameters;
    }
    
    /*
     Is this tested?
     I think it can be better done by repeated vector dot product
     */
    vector<vector<double>> dot(vector<vector<double>>& A, vector<vector<double>>& B) {
        vector<vector<double>> C;
        for (int i = 0; i < A.size(); i++) {
            vector<double> s;
            for (int j = 0; j < A[i].size(); j++)
                s.push_back(A[i][j] * B[i][j]);
            C.push_back(s);
        }
        return C;
    }
    
    /*
     Linear forward should output a vector of pre-activations, not a 2D vector of activations
     */
    vector<vector<double>> linearForwardZ(vector<vector<double>>& A, vector<vector<double>>& W, vector<vector<double>>& b) {
        /*Arguments:
         A -- activations from previous layer (or input data): (size of previous layer, number of examples)
         W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
         b -- bias vector, numpy array of shape (size of the current layer, 1)
         
         Returns:
         Z -- the input of the activation function, also called pre-activation parameter
         */
        
        /*
         Please assert the dimensions of A and W before dot product
         */
        vector<vector<double>> Z = dot(A, W);
        for (int i = 0; i < Z.size(); i++)
            for(int j=0; j<Z[i].size();j++)
                Z.at(i).at(j) = Z.at(i).at(j) + b.at(i).at(0);
        return Z;
    }
    
    /*
     What is the difference between this and the previous?
     */
    unordered_map<string, vector<vector<double>>> linearForwardCache(vector<vector<double>>& A, vector<vector<double>>& W, vector<vector<double>>& b) {
        /*Arguments:
         A -- activations from previous layer (or input data): (size of previous layer, number of examples)
         W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
         b -- bias vector, numpy array of shape (size of the current layer, 1)
         
         Returns:
         cache -- hashmap containing A, W, and b used in backprop
         */
        unordered_map<string, vector<vector<double>>> cache;
        cache.insert(make_pair("A" , A));
        cache.insert(make_pair("W", W));
        cache.insert(make_pair("b", b));
        return cache;
    }
    
    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }
    
    double relu(double z) {
        return z<0 ? 0 : z;
    }
    
    vector<int> shape(vector<vector<double>>& A) {
        vector<int> s,w;
        bool consistent = true;
        s.push_back(A.size());
        for (int i = 1; i < s[0]; i++) {
            if (A[i].size() != A[i - 1].size()) {
                cout << "shape is not consistent";
                consistent = false;
            }
            
        }
        s.push_back(A[0].size());
        if (consistent)
            return s;
        else {
            w.push_back(-1);
            return w;
        }
    }
    
    void printShape(vector<int> & A) {
        cout << "Shape  is: ( " << A[0] << " , " << A[1] << " )" << endl;
    }
    
    //activation 0 for sigmoid, 1 for relu
    vector<vector<double>> linearActivationForwardA(vector<vector<double>> &A_prev, vector<vector<double>>& W, vector<vector<double>>& b,int activation) {
        vector<vector<double>> Z = linearForwardZ(A_prev, W, b);
        if (!activation) {
            for (int i = 0; i < Z.size(); i++) {
                for (int j = 0; j < Z[i].size(); j++)
                    Z.at(i).at(j) = sigmoid(Z.at(i).at(j));
            }
        }
        else if (activation) {
            for (int i = 0; i < Z.size(); i++) {
                for (int j = 0; j < Z[i].size(); j++)
                    Z.at(i).at(j) = relu(Z.at(i).at(j));
            }
        }
        else
            cout << "wrong activation"<<endl;
        return Z;
    }
    
    //activation 0 for sigmoid, 1 for relu
    vector<unordered_map<string, vector<vector<double>>>> linearActivationForwardCache(vector<vector<double>>& A_prev, vector<vector<double>>& W, vector<vector<double>>& b, int activation) {
        vector<vector<double>> Z = linearForwardZ(A_prev, W, b);
        vector<unordered_map<string, vector<vector<double>>>> cache;
        unordered_map<string, vector<vector<double>>> linear_cache = linearForwardCache(A_prev, W, b), activation_cache;
        if (!activation) {
            for (int i = 0; i < Z.size(); i++)
                Z.at(i).at(0) = sigmoid(Z.at(i).at(0));
            activation_cache.insert(make_pair("Z", Z));
        }
        else if (activation) {
            for (int i = 0; i < Z.size(); i++)
                Z.at(i).at(0) = relu(Z.at(i).at(0));
            activation_cache.insert(make_pair("Z", Z));
        }
        cache.push_back(linear_cache);
        cache.push_back(activation_cache);
        return cache;
    }
    
    vector<vector<double>> LModelForwardAL(vector<vector<double>>& X, unordered_map<string, vector<vector<double>>>& parameters) {
        vector<vector<double>> A = X, A_prev, W ,b,AL;
        int len = int(parameters.size());
        for (int i = 1; i < len; i++) {
            A_prev = A;
            W = parameters['W' + to_string(i)];
            b = parameters['b' + to_string(i)];
            A = linearActivationForwardA(A_prev, W, b , 1);
        }
        W = parameters['W' + to_string(len)];
        b = parameters['b' + to_string(len)];
        AL = linearActivationForwardA(A, W, b, 0);
        return AL;
    }
    
    vector<vector<unordered_map<string, vector<vector<double>>>>>& LModelForwardCaches(vector<vector<double>>& X, unordered_map<string, vector<vector<double>>>& parameters) {
        vector<unordered_map<string, vector<vector<double>>>>cache;
        vector<vector<unordered_map<string, vector<vector<double>>>>> caches;
        vector<vector<double>> A = X, A_prev, W, b, AL;
        int len = parameters.size();
        for (int i = 1; i < len; i++) {
            A_prev = A;
            W = parameters['W' + to_string(i)];
            b = parameters['b' + to_string(i)];
            A = linearActivationForwardA(A_prev, W, b, 1);
            cache = linearActivationForwardCache(A_prev, W, b, 1);
            caches.push_back(cache);
        }
        W = parameters['W' + to_string(len)];
        b = parameters['b' + to_string(len)];
        AL = linearActivationForwardA(A, W, b, 0);
        cache = linearActivationForwardCache(A, W, b, 0);
        caches.push_back(cache);
        return caches;
    }
    
    double computeCost(vector<vector<double>> AL, vector<vector<double>> Y) {
        int m = Y[0].size();
        double cost = 0;
        for (int i = 0; i < m; i++)
            cost += Y[0][i] * log(AL[i][0]) + (1 - Y[0][i]) * log(1 - AL[i][0]);
        cost *= (-1.0 / m);
        return cost;
    }
    
    vector<vector<double>> transpose(vector<vector<double>>& A) {
        vector<vector<double>> B;
        for (int i = 0; i < A[i].size(); i++)
        {
            vector<double> s;
            for (int j = 0; j < A.size(); j++) {
                s.push_back(A[j][i]);
            }
            B.push_back(s);
        }
        B.pop_back();
        return B;
    }
    
    vector<vector<vector<double>>> linearBackward(vector<vector<double>> dZ, unordered_map<string, vector<vector<double>>>& cache) {
        vector<vector<double>> W = cache["W"];
        vector<vector<double>>b = cache["b"];
        vector<vector<double>> A_prev = cache["A"];
        int m = A_prev[0].size();
        vector<vector<double>> dA_prev, dW, db, temp;
        vector<vector<vector<double>>>output;
        A_prev = transpose(A_prev);
        W = transpose(W);
        dW = dot(dZ, A_prev);
        // if it gives an error then use push back
        for (int i = 0; i < db.size(); i++)
            for (int j = 0; j < db[i].size(); j++)
                temp[i][j] = 0;
        for (int i = 0; i < db.size(); i++)
            for (int j = 0; j < db[i].size(); j++)
                temp[0][j] = temp[0][j] + db[i][j];
        dA_prev = dot(W, dZ);
        for (int i = 0; i < dW.size(); i++)
            for (int j = 0; j < dW[i].size(); j++) {
                dW[i][j] *= (1 / m);
                db[i][j] *= (1 / m);
            }
        db = temp;
        
        output.push_back(dA_prev);
        output.push_back(dW);
        output.push_back(db);
        return output;
    }
    
    vector<vector<double>> reluBackward(vector<vector<double>> dA, unordered_map<string, vector<vector<double>>> cache) {
        vector<vector<double>> dZ;
        //coping matrix
        for (int i = 0; i < dA.size(); i++) {
            vector<double> s;
            for (int j = 0; j < dA[i].size(); j++)
                s.push_back(dA[i][j]);
            dZ.push_back(s);
        }
        vector<vector<double>> Z = cache["Z"];
        for (int i = 0; i < dA.size(); i++) {
            for (int j = 0; j < dA[i].size(); j++)
                if (Z[i][j] < 0)
                    dZ[i][j] = 0;
        }
        return dZ;
    }
    
    vector<vector<double>>& sigmoidBackward(vector<vector<double>> dA, unordered_map<string, vector<vector<double>>> cache) {
        vector<vector<double>> dZ;
        vector<vector<double>> Z = cache["Z"];
        for (int i = 0; i < Z.size(); i++)
            for(int j=0; j<Z[i].size();j++)
                Z.at(i).at(j) = sigmoid(Z.at(i).at(j));
        
        for (int i = 0; i < dA.size(); i++)
        {
            vector<double> s;
            for (int j = 0; j < dA[i].size(); j++)
                s.push_back(dA[i][j]* Z[i][j]*(1-Z[i][j]));
        }
        return dZ;
    }
    
    vector<vector<vector<double>>>& linearActivationBackward(vector<vector<double>> dA, vector<unordered_map<string, vector<vector<double>>>>& cache, string activation) {
        unordered_map<string, vector<vector<double>>> linear_cache = cache[0], activation_cache = cache[1];
        vector<vector<vector<double>>>output,temp;
        vector<vector<double>> dZ, dA_prev, dW, db;
        if (activation.compare("relu") == 0) {
            dZ = reluBackward(dA, activation_cache);
            temp = linearBackward(dZ, linear_cache);
            dA_prev = temp[0];
            dW = temp[1];
            db = temp[2];
        }
        else if (activation.compare("sigmoid") == 0) {
            dZ = sigmoidBackward(dA, activation_cache);
            temp = linearBackward(dZ, linear_cache);
            dA_prev = temp[0];
            dW = temp[1];
            db = temp[2];
        }
        output.push_back(dA_prev);
        output.push_back(dW);
        output.push_back(db);
        return output;
    }
    
    unordered_map<string, vector<vector<double>>>& LModelBackward(vector<vector<double>>& AL, vector<vector<double>> & Y, vector<vector<unordered_map<string, vector<vector<double>>>>>& caches) {
        unordered_map<string, vector<vector<double>>> grads;
        int len = caches.size();
        int m = AL[0].size();
        // HEREEEEEEEEEEEEEEEEE you have to reshape Y to be the same shape as AL
        vector<vector<double>> dAL = AL, dA_prev_temp, dW_temp, db_temp;
        for (int i = 0; i < dAL.size(); i++)
            for (int j = 0; j < dAL[i].size(); j++)
                dAL[i][j] = -((Y[i][j] / AL[i][j]) - ((1 - Y[i][j]) / (1 - AL[i][j])));
        vector<unordered_map<string, vector<vector<double>>>> currentCache = caches[len - 1];
        vector<vector<vector<double>>> temp = linearActivationBackward(dAL, currentCache, "sigmoid");
        grads["dA" + to_string(len - 1)] = temp[0];
        grads["dW" + to_string(len)] = temp[1];
        grads["db" + to_string(len)] = temp[2];
        for(int i = len - 1; i >= 0; i++) {
            currentCache = caches[i];
            temp = linearActivationBackward(grads["dA" + to_string(i + 1)], currentCache, "relu");
            grads["dA" + to_string(i)] = temp[0];
            grads["dW" + to_string(i+1)] = temp[1];
            grads["db" + to_string(i + 1)] = temp[2];
        }
        return grads;
    }
    
    unordered_map<string, vector<vector<double>>>& updateParameters(unordered_map<string, vector<vector<double>>>& params, unordered_map<string, vector<vector<double>>>& grads, double learningRate) {
        unordered_map<string, vector<vector<double>>> parameters = params;
        int len = parameters.size();
        for (int i = 0; i < len; i++) {
            //parameters["W" + to_string(i + 1)] = parameters["W" + to_string(i + 1)] - learningRate * grads["dW" + to_string(i + 1)];
            //parameters["b" + to_string(i + 1)] = parameters["b" + to_string(i + 1)] - learningRate * grads["db" + to_string(i + 1)];
            for(int j = 0; j< parameters["W" + to_string(i + 1)].size(); j++)
                for (int k = 0; k < parameters["W" + to_string(i + 1)][j].size(); k++) {
                    parameters["W" + to_string(i + 1)][j][k] = parameters["W" + to_string(i + 1)][j][k] - learningRate * grads["dW" + to_string(i + 1)][j][k];
                }
            for (int j = 0; j < parameters["b" + to_string(i + 1)].size(); j++)
                for (int k = 0; k < parameters["b" + to_string(i + 1)][j].size(); k++) {
                    parameters["b" + to_string(i + 1)][j][k] = parameters["b" + to_string(i + 1)][j][k] - learningRate * grads["db" + to_string(i + 1)][j][k];
                }
        }
        return parameters;
    }
    
    // CREATING THE ACTUAL MODEL
    // X should be an array of images and images has 3 dimensions so X must be a 4d vector, the first to determine each training expamle and the remmaining three for the image. To do so now requries alot of modifications in the code so will do this another time.
    unordered_map<string, vector<vector<double>>> LLayerModel(vector<vector<vector<double>>>& X, vector<vector<double>>& Y, vector<int>& layers_dims, double learningRate = 0.0075,int numIterations = 30,bool print_cost = false) {
        srand(1);
        double cost;
        vector<double> costs;
        vector<vector<double>> AL;
        vector<vector<unordered_map<string, vector<vector<double>>>>> caches;
        unordered_map<string, vector<vector<double>>>grads, parameters = initializeParametersDeep(layers_dims);
        for (int i = 0; i < numIterations; i++) {
            AL = LModelForwardAL(X[i], parameters);
            caches = LModelForwardCaches(X[i], parameters);
            cost = computeCost(AL, Y);
            grads = LModelBackward(AL, Y, caches);
            parameters = updateParameters(parameters, grads, learningRate);
            
            cout << "Cost after iteration " << i+1 << "is: "<< cost<<endl;
        }
        return parameters;
    }
    
    // same goes for this X it must be an images of 3d vectors.
    bool predict(vector<vector<double>> &X, int y, unordered_map<string, vector<vector<double>>> parameters) {
        // Unused
        // int n = int(parameters.size());
        vector<vector<double>> p = LModelForwardAL(X, parameters);
        int prediction;
        // test by printing p and look where is the value of the sigmoid unit, then use the index to comapre it with y but for now let's suppose that the value of the sigmoid unit lies in p[0][0]
        if (p[0][0] > 0.5)
            prediction = 1;
        else
            prediction = 0;
        
        return bool(prediction);
        // I didn't learn the accuracy and this method should be to predict multipule things but I make it predict only one thing so maybe I could use two methods one to calculate accuracy and another for predictoin a specific image
    }
    
    // what is left? data preprocessing like: loading data, convertion from image to array, normalization, plotting images, plotting images as a 2d array.
    
};
