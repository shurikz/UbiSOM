#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip> 
#include <vector> 

using namespace std;

class UbiSOM
{
public:
	UbiSOM(const int entersNumber, const int height, const int width, const double min = -0.5, const double max = 0.5,
		const int topol = RECT, const int windowLength = 2000, const float eta0 = 0.1, const float etaF = 0.08, const float sigma0 = 0.6, const float sigmaF = 0.2,
		const float beta = 0.7);
	~UbiSOM();
	void trainMap(const string&,bool);
	istream& trainMap(istream&);
	void trainMap(const double*);
	int findBMU(const double*);
	void saveMap(const char*);
	void saveMap(const string&);
	void getNormalizeVector(const string&);
	void getNormalizeVector(const char*);
	void normalizeDatabyMinMax(double* data);
	void initWeights();
	const enum cell { RECT, HEXA };
	double getWidth() { return mapWidth; }
	double getHeight() { return mapHeight; }
	double** getUmatrix();

private:
	int windowsNumber;

	double** map;
	int mapWidth;
	int mapHeight;
	int neuronNumber;
	double diag;		//”¡–¿“‹, ≈—À» Õ≈ œŒ“–≈¡”≈“—ﬂ!!!!!!!!!!!!!!!!!!!!
	int variableNumber;
	short cellType;


	double* quantizationError;
	double* neuronUtility;
	int *lastUpdate;

	double bigSigma;
	int iteration;
	int T;
	float eta0;
	float etaF;
	float sigma0;
	float sigmaF;
	double* normalizeVectorMAX;
	double* normalizeVectorMIN;
	float beta;

	void deleteFirstUpdate();
	void computeSigmaEta(double&, double&);

	bool isOrderState;
	int whichLastBigger;
	int countWhichBigger;

	void changeState();
	int* neuronCoord(const int&);
	double rectDist(const int&, const int&);
	double hexDist(const int&, const int&);
	double neuronDist(const int&, const int&);


	double neighborhoodFun(const int&, const int&, const double&);
	//double(*neighborhoodFun)(const double&, const double&, const double&);
	double computeNeuronUtility();
	double evklDist(const double*, const double *);

	double driftFunction();
	double lastDriftValue;
	void countBigSigma();
	void randomInitialization();
	void shiftArray();
	void checkState();
};

