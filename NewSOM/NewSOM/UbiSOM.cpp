#include "UbiSOM.h"

ofstream qeHes;
ofstream clFile;

template<class T>
void setArray(T*arr, int size, T val)
{
	for (int i = 0; i < size; i++)
		arr[i] = val;
}

UbiSOM::UbiSOM(const int entersNumber, const int height, const int width, const double min, const double max,
	const int topol /*=RECT*/, const int windowLength /*= 2000*/, const float eta0 /*= 0.1*/,
	const float etaF /*= 0.08*/, const float sigma0/* = 0.6*/, const float sigmaF /*= 0.2*/,
	const float beta /*= 0.7*/) :
	variableNumber(entersNumber), mapHeight(height), mapWidth(width), cellType(topol),
	T(windowLength), eta0(eta0), etaF(etaF), sigma0(sigma0), sigmaF(sigmaF), beta(beta)
{
	neuronNumber = mapWidth*mapHeight;
	map = new double*[neuronNumber];
	map[0] = new double[variableNumber*neuronNumber];
	for (size_t i = 1; i<neuronNumber; i++)
		map[i] = map[i - 1] + variableNumber;

	quantizationError = new double[windowLength];
	neuronUtility = new double[windowLength];

	lastUpdate = new int[neuronNumber];
	setArray(lastUpdate, neuronNumber, (-1)*T);

	if (cellType == RECT) diag = sqrt(pow(mapWidth - 1, 2) + pow(mapHeight - 1, 2));
	else diag = neuronDist(0, neuronNumber - 1);

	normalizeVectorMAX = new double[entersNumber];
	setArray(normalizeVectorMAX, variableNumber, max);

	normalizeVectorMIN = new double[entersNumber];
	setArray(normalizeVectorMIN, variableNumber, min);


	countBigSigma();

	whichLastBigger = -2;
	countWhichBigger = 0;
	iteration = 0;
	isOrderState = true;
	randomInitialization();

}

UbiSOM::~UbiSOM()
{
	delete[]map[0];
	delete[] map;
	delete[] normalizeVectorMAX;
	delete[] normalizeVectorMIN;
	delete[] quantizationError;
	delete[] lastUpdate;
	delete[] neuronUtility;
	qeHes.close();
}

double UbiSOM::evklDist(const double* mapWeights, const double* data)
{
	double sum = 0;
	for (int i = 0; i < variableNumber; i++)
		sum += pow(mapWeights[i] - data[i], 2);
	return sqrt(sum);
}

double UbiSOM::computeNeuronUtility()
{
	int num = 0;
	for (int i = 0; i < neuronNumber; i++)
	{
		if (lastUpdate[i] != (-1)*T)
			num++;
	}
	return (double)num / neuronNumber;
}

int UbiSOM::findBMU(const double* data)
{
	int index = 0;
	double minDist = evklDist(map[0], data);
	for (int i = 1; i < neuronNumber; i++)
	{
		double dist = evklDist(map[i], data);
		if (dist < minDist)
		{
			minDist = dist;
			index = i;
		}
	}
	quantizationError[isOrderState ? iteration /*- 1*/ : T - 1] = minDist / bigSigma;//???????
	qeHes << quantizationError[isOrderState ? iteration/* - 1*/ : T - 1] << ' ';// << ' '<< neuronUtility[isOrderState ? iteration - 1 : T - 1] << endl;
	return index;
}

int* UbiSOM::neuronCoord(const int& neuron)
{
	int*  ans = new int[2];
	ans[0] = neuron / mapHeight;//x координата i-го нейрона;
	ans[1] = neuron %mapHeight;//y координата i-го нейрона;
	return ans;
}

double UbiSOM::rectDist(const int& neuron, const int& winingNeuron)
{
	//double** coord=neuronCoord(neuron,winingNeuron);
	//double ans= sqrt(pow(coord[0][0] - coord[1][0], 2) + pow(coord[0][1] - coord[1][1], 2));//(x0-x1)^2+(y0-y1)^2
	int * nCoord = neuronCoord(neuron);
	int* wCoord = neuronCoord(winingNeuron);
	//double ans = sqrt(pow(nCoord[0] - wCoord[0], 2) + pow(nCoord[1] - wCoord[1], 2));
	double ans = 0;

	ans = abs(nCoord[0] - wCoord[0]) + abs(nCoord[1] - wCoord[1]);
	//ans = sqrt(pow(nCoord[0] - wCoord[0], 2) + pow(nCoord[1] - wCoord[1], 2));
	delete[] nCoord;
	delete[] wCoord;
	return ans;
}

double UbiSOM::hexDist(const int& neuron, const int& winingNeuron)
{
	//double**coord = neuronCoord(neuron, winingNeuron);
	int* nCoord = neuronCoord(neuron);
	double xN = nCoord[0], yN = nCoord[1];
	int* wCoord = neuronCoord(winingNeuron);
	double xW = wCoord[0], yW = wCoord[1];

	if ((int)yN % 2 != 0)
		xN += 0.5;
	if ((int)yW % 2 != 0)
		xW += 0.5;
	yN *= sqrt(0.75);
	yW *= sqrt(0.75);
	//double ans = sqrt(pow(xN - xW, 2) + pow(yN - yW, 2));//(x0-x1)^2+(y0-y1)^2
	double ans = 0;

	ans = sqrt(pow(xN - xW, 2) + pow(yN - yW, 2));//(x0-x1)^2+(y0-y1)^2

	delete[] nCoord;
	delete[] wCoord;

	return ans;
}

double UbiSOM::neuronDist(const int& neuron, const int& winingNeuron)
{
	if (cellType == HEXA)return hexDist(neuron, winingNeuron);
	else return rectDist(neuron, winingNeuron);
}

double UbiSOM::neighborhoodFun(const int& neuron, const int& winingNeuron, const double& sigma)
{
	return exp((-1)*pow(neuronDist(neuron, winingNeuron) / (sigma*diag), 2));
}

void UbiSOM::shiftArray()
{
	int i = T- 1;
	double val = quantizationError[i],
	val2 = neuronUtility[i];
	for (i; i>=0; i--)
	{
		quantizationError[i] += val;
		val = quantizationError[i] - val;
		quantizationError[i] = quantizationError[i] - val;

		neuronUtility[i] += val2;
		val2 = neuronUtility[i] - val2;
		neuronUtility[i] = neuronUtility[i] - val2;
	}
	/*quantizationError[0] = val;
	neuronUtility[0] = val2;
*/
	//a[size-1] = (-1)*size - 1;

}

int findMin(int* arr, const int&len, const int& T)
{
	int a = 0;
	for (int i = 0; i < len; i++)
	{
		if ((arr[i] < a) && T != arr[i])
			a = arr[i];
	}
	return a;
}

void UbiSOM::deleteFirstUpdate()
{
	for (int i = 0; i < neuronNumber; i++)
	{
		if (lastUpdate[i] < 0 && lastUpdate[i] * (-1) <= iteration && lastUpdate[i] * (-1) < T)
		{
			lastUpdate[i] = -T;
			continue;
		}
		if (lastUpdate[i] == 0 && iteration == 0)
		{
			lastUpdate[i] = -T;
			continue;
		}
	}
	int a = findMin(lastUpdate, neuronNumber, -T);

}

void UbiSOM::trainMap(const string& path,bool isNorm)
{
	string newPath=path;
	if (!isNorm)
	{
		newPath = "norm" + path;
		ofstream normFile(newPath,ios_base::out);
		ifstream f;
		getNormalizeVector(path);

		f.open(path);
		double* data = new double[variableNumber];
		int j = 0;
		while (!normFile.eof())
		{
			for (int i = 0; i < variableNumber; i++)
						f >> data[i];
			normalizeDatabyMinMax(data);

			for (int i = 0; i < variableNumber; i++)
					normFile << data[i]<<' ';
				normFile << endl;
			cout <<"Vectors normilized: "<< ++j << endl;

		}
		f.close();
		normFile.close();
		delete[]data;
	}
	ifstream file;
	getNormalizeVector(newPath);
	initWeights();
	file.open(newPath);
	trainMap(file);
	file.close();
}


istream& UbiSOM::trainMap(istream& input)
{
	double* data = new double[variableNumber];
	int j = 0;
	while (input.good())
	{
		for (int i = 0; i < variableNumber; i++)
				input >> data[i];
		trainMap(data);

		cout << "Vectors train: " << ++j << endl;
	}
	delete[]data;
	return input;
}

void UbiSOM::trainMap(const double* data)
{
	double etaI = 0, sigmaI = 0, neighborhood = 0;
	if (!isOrderState)
	{
		shiftArray();
		deleteFirstUpdate();
	}
	int BMU = findBMU(data);
	computeSigmaEta(sigmaI, etaI);
	qeHes << etaI << ' ' << sigmaI << ' ';

	for (int i = 0; i < neuronNumber; i++)
	{
		neighborhood = neighborhoodFun(i, BMU, sigmaI);
		if (neighborhood > 0.01)
		{
			for (int j = 0; j < variableNumber; j++)
				map[i][j] += etaI*neighborhood*(data[j] - map[i][j]);
			lastUpdate[i] = iteration;
		}
	}
	neuronUtility[isOrderState ? iteration/*-1*/ : T - 1] = computeNeuronUtility();
	qeHes << neuronUtility[isOrderState ? iteration/*-1*/ : T - 1] << endl;


	iteration++;
	checkState();
	
}

void UbiSOM::checkState()
{
	if (isOrderState)
	{
		if (iteration == T)
		{
			changeState();
		}
	}
	else
	{

		if (/*countWhichBigger >= T-1||*/ countWhichBigger >= T)
		{
			changeState();
		}
		if (iteration == T)
		{
			for (int i = 0; i < neuronNumber; i++)
			{
				if (lastUpdate[i] > 0)
					lastUpdate[i] *= (-1);
			}
			if (whichLastBigger == -2)whichLastBigger = -1;//за один цикл лернинг не было случаем, когда дрифт больше оригина
			iteration = 0;
		}
	}
}

void UbiSOM::computeSigmaEta(double& sigmaI, double& etaI)
{
	double arg = 0;
	if (isOrderState)
	{
		arg = (double)iteration / (T - 1);
		sigmaI = sigma0*pow(sigmaF / sigma0, arg);
		etaI = eta0*pow(etaF / eta0, arg);
	}
	else
	{
		arg = driftFunction();

		if (arg < lastDriftValue)
		{
			sigmaI = (iteration == 0) && (whichLastBigger == -2) ? sigmaF : (arg*sigmaF) / lastDriftValue;
			etaI = (iteration == 0) && (whichLastBigger == -2) ? etaF : (arg*etaF) / lastDriftValue;
			if (countWhichBigger)
			{
				countWhichBigger = 0;
				whichLastBigger = -3;
			}
		}
		else
		{
			sigmaI = sigmaF;
			etaI = etaF;
			if (whichLastBigger == ((iteration - 1 == -1/*0*/) ? T - 1 : iteration - 1))
			{
				whichLastBigger = iteration;
				countWhichBigger++;
			}
			else
			{
				whichLastBigger = iteration;
				countWhichBigger = 1;
			}
		}
	}
}

void UbiSOM::changeState()
{
	if (isOrderState)
	{
		lastDriftValue = driftFunction();
		for (int i = 0; i < neuronNumber; i++)
		{
			if (lastUpdate[i] != -T)lastUpdate[i] *= -1;
			//if (lastUpdate[i] ==0)lastUpdate[i] =lastVal;
		}
		cout << "--------------------------LEARNING STATE--------------------------";
		iteration = 0;
		countWhichBigger = 0;
		whichLastBigger = -2;
		isOrderState = false;
	}
	else
	{
		cout << "--------------------------ORDER STATE--------------------------";
		setArray(lastUpdate, neuronNumber, -T);

		iteration = 0;
		countWhichBigger = 0;
		whichLastBigger = -2;
		isOrderState = true;
	}
}

double UbiSOM::driftFunction()
{
	double qe = 0;
	for (int i = 0; i <T; i++)
		qe += quantizationError[i];
	qe /= T;
	double he = 0;
	for (int i = 0; i < T; i++)
	{
		he += neuronUtility[i];
	}
	he /= T;
	return beta*qe + (1 - beta)*(1 - he);
}

void UbiSOM::initWeights()
{
	countBigSigma();
	randomInitialization();
}

void UbiSOM::normalizeDatabyMinMax(double* data)
{
	for (int i = 0; i < variableNumber; i++)
	{
		if (normalizeVectorMAX[i] != 0)
			data[i] = (data[i] - normalizeVectorMIN[i]) / (normalizeVectorMAX[i] - normalizeVectorMIN[i]);
		else
			data[i] = 0;
	}
}

void UbiSOM::randomInitialization()
{
	srand(time(NULL));
	for (int i = 0; i < neuronNumber; i++)
		for (int j = 0; j < variableNumber; j++)
			map[i][j] = (double)(rand()) / RAND_MAX*(normalizeVectorMAX[j] - normalizeVectorMIN[j]) + normalizeVectorMIN[j];
}

void UbiSOM::saveMap(const string &path)
{
	saveMap(path.c_str());
}

void UbiSOM::saveMap(const char* path)
{
	char s[30];
	strcpy_s(s, path);
	strcat_s(s,".cod");
	ofstream fout(s, ios_base::out);
	fout << variableNumber << " ";
	if (cellType == RECT)fout << "rect";
	else fout << "hexa";
	fout << " " << mapWidth << " " << mapHeight << " ";
	fout << "gaussian";
	fout << endl;

	fout << "#n ";

	for (int i = 0; i < variableNumber; i++)
		fout << '-' << " ";
	fout << endl;

	for (int i = 0; i < mapHeight; i++)
	{
		for (int j = 0; j < mapWidth; j++)
		{
			for (int k = 0; k < variableNumber; k++)
				fout << map[mapHeight*j + i][k] << ' ';
			fout << endl;
		}
	}

	fout.close();
}

void UbiSOM::getNormalizeVector(const string& path)
{
	getNormalizeVector(path.c_str());
}

double** UbiSOM::getUmatrix()
{
	double*** M = new double**[variableNumber];
	M[0] = new double*[mapHeight*variableNumber];
	M[0][0] = new double[neuronNumber*variableNumber];

	for (int i = 1; i < variableNumber; i++)
		M[i] = M[i - 1] + mapHeight;

	for (int j = 1; j < mapHeight*variableNumber; j++)
		M[0][j] = M[0][j - 1] + mapWidth;

	for (int i = 0; i < variableNumber; i++)
		for (int j = 0; j < mapHeight; j++)
			for (int k = 0; k<mapWidth; k++)
				M[i][j][k] = map[k*mapHeight + j][i];

	int HU = 2 * mapHeight - 1, WU = 2 * mapWidth - 1;

	double**Umatr = new double*[HU];
	Umatr[0] = new double[HU*WU];
	for (int i = 1; i < HU; i++)
		Umatr[i] = Umatr[i - 1] + WU;

	for (int i = 0; i < HU; i++)
		for (int j = 0; j < WU; j++)
			Umatr[i][j] = 0;

	double dx = 0, dy = 0, dz1 = 0, dz2 = 0;
	if (cellType == RECT) {
		for (int i = 0; i < mapHeight; i++)
			for (int j = 0; j < mapWidth; j++)
			{
				if (j < mapWidth - 1)
				{
					for (int k = 0; k < variableNumber; k++)
						dx += pow(M[k][i][j] - M[k][i][j + 1], 2);
					Umatr[2 * i][2 * j + 1] = sqrt(dx);
				}
				if (i < mapHeight - 1)
				{
					for (int k = 0; k < variableNumber; k++)
						dy += pow(M[k][i][j] - M[k][i + 1][j], 2);
					Umatr[2 * i + 1][2 * j] = sqrt(dy);
				}
				if (i < (mapHeight - 1) && j < (mapWidth - 1))
				{
					for (int k = 0; k < variableNumber; k++)
					{
						dz1 += pow(M[k][i][j] - M[k][i + 1][j + 1], 2);
						dz2 += pow(M[k][i][j + 1] - M[k][i + 1][j], 2);
					}
					Umatr[2 * i + 1][2 * j + 1] = (sqrt(dz1) + sqrt(dz2)) / (2 * sqrt(2));
				}
				dx = dz1 = dz2 = dy = 0;
			}
	}
	else
	{
		for (int i = 0; i < mapHeight; i++)
			for (int j = 0; j < mapWidth; j++)
			{
				if (j < mapWidth - 1)
				{
					for (int k = 0; k < variableNumber; k++)
						dx += pow(M[k][i][j] - M[k][i][j + 1], 2);
					Umatr[2 * i][2 * j + 1] = sqrt(dx);
				}
				if (i < mapHeight - 1)
				{
					for (int k = 0; k < variableNumber; k++)
						dy += pow(M[k][i][j] - M[k][i + 1][j], 2);
					Umatr[2 * i + 1][2 * j] = sqrt(dy);
				}

				if ((i % 2 != 0) && i < (mapHeight - 1) && j < (mapWidth - 1))
				{
					for (int k = 0; k < variableNumber; k++)
						dz1 += pow(M[k][i][j] - M[k][i + 1][j + 1], 2);
					Umatr[2 * i + 1][2 * j + 1] = sqrt(dz1);
				}
				else if ((i % 2 == 0) && j > 0)
				{
					for (int k = 0; k < variableNumber; k++)
						dz1 += pow(M[k][i][j] - M[k][i + 1][j - 1], 2);
					Umatr[2 * i + 1][2 * j - 1] = sqrt(dz1);
				}
				dx = dz1 = dz2 = dy = 0;
			}
	}
	delete[] M[0][0];
	delete[]M[0];
	delete[] M;

	double mean = 0;
	int nX, nY, countNeighboor = 0;
	if (cellType == RECT) {
		for (int i = 0; i < HU; i += 2)
			for (int j = 0; j < WU; j += 2)
			{
				for (int k = -1; k < 2; k += 2)
				{
					nY = i + k;
					if (nY >= 0 && nY < HU)
					{
						mean += Umatr[nY][j];
						countNeighboor++;
					}
				}
				for (int k = -1; k < 2; k += 2)
				{
					nX = j + k;
					if (nX >= 0 && nX < WU)
					{
						mean += Umatr[i][nX];
						countNeighboor++;
					}
				}
				Umatr[i][j] = mean / countNeighboor;
				countNeighboor = 0;
				mean = 0;
			}
	}
	else
	{
		bool isOdd = false;
		for (int i = 0; i < HU; i += 2)
		{
			for (int j = 0; j < WU; j += 2)
			{
				for (int k = -1; k < 2; k += 2)
				{
					nY = i + k;
					if (nY >= 0 && nY < HU)
					{
						mean += Umatr[nY][j];
						countNeighboor++;
					}
				}
				for (int k = -1; k < 2; k += 2)
				{
					nX = j + k;
					if (nX >= 0 && nX < WU)
					{
						mean += Umatr[i][nX];
						countNeighboor++;
					}
				}

				if (isOdd)
				{
					nY = i - 1;
					nX = j + 1;
					if ((nX >= 0 && nX < WU) && (nY >= 0 && nY < HU))
					{
						mean += Umatr[nY][nX];
						countNeighboor++;
					}
					nY = i + 1;
					nX = j + 1;
					if ((nX >= 0 && nX < WU) && (nY >= 0 && nY < HU))
					{
						mean += Umatr[nY][nX];
						countNeighboor++;
					}
				}
				else
				{
					nY = i + 1;
					nX = j - 1;
					if ((nX >= 0 && nX < WU) && (nY >= 0 && nY < HU))
					{
						mean += Umatr[nY][nX];
						countNeighboor++;
					}
					nY = i - 1;
					nX = j - 1;
					if ((nX >= 0 && nX < WU) && (nY >= 0 && nY < HU))
					{
						mean += Umatr[nY][nX];
						countNeighboor++;
					}

				}

				Umatr[i][j] = mean / countNeighboor;
				countNeighboor = 0;
				mean = 0;
			}
			if (isOdd)isOdd = false;
			else isOdd = true;
		}
	}
	return Umatr;
}

void UbiSOM::countBigSigma()
{
	bigSigma = 0;
	for (int i = 0; i < variableNumber; i++)
		bigSigma += pow((normalizeVectorMAX[i] - normalizeVectorMIN[i])*sqrt(variableNumber), 2);
	bigSigma = sqrt(bigSigma);
}

void UbiSOM::getNormalizeVector(const char* path)
{
	ifstream fin(path);
	double number = 0;
	for (int i = 0; i < variableNumber; i++)
	{
		fin >> number;
		normalizeVectorMAX[i] = number;
		normalizeVectorMIN[i] = number;
	}
	while (!fin.eof())
	{
		for (int i = 0; i < variableNumber; i++)
		{
			fin >> number;
			if (number > normalizeVectorMAX[i])
				normalizeVectorMAX[i] = number;
			if (number < normalizeVectorMIN[i])
				normalizeVectorMIN[i] = number;
		}
	}
	countBigSigma();
	string q = string(path);
	q = q.substr(0, q.length() - 4);

	if (qeHes.is_open())qeHes.close();
	qeHes.open(q + "qeHe.txt", ios_base::out);
	fin.close();
}

