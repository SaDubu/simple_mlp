#include "model.h"
#include "cmatrix.h"
#include "calcmat.h"
#include "idxpasing.h"
#include "cio.h"
#include "pathchecker.h"
#include "rmDir.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <stddef.h>
#include <random>
#include <limits>
#include <sys/stat.h>
#include <thread>

const std::string weightDirPath = "saveModel/weights";
const std::string biasesDirPath = "saveModel/biases";

const std::string bestDirPath = "Best_Accuary";

const std::string numOfHiddenForOuput = "number of hiddenLayer";

const std::string sizeOfHiddenForOuput = "size of hiddenLayer";

const std::string numOfEpoch = "number of epoch";

const int BATCH_SIZE_ONE = 1;

const int BATCH_SIZE_FIVE_HUNDRED = 100;

CIo inOut;
CCalc calcMat;
CParsing parsingFile;	
CPathChecker pathCecker;
CRMDir rmDir;

double bestAcc = 0.0;
bool bestSaveOn = false;

std::string rinl = "";
std::vector<int> tokens;
std::vector<std::string> mun;
std::vector<std::string> datas;

int epoch = 0;
double duration = 0.0;

std::string optimizerName = "";

void CModel::setS(std::string a) {
	std::vector<int>().swap(tokens);
	rinl = a;
	std::istringstream iss(rinl);
	
	std::string token;
	while (iss >> token) {
		datas.emplace_back(token);
		try {
			if (token.find(".") != std::string::npos) {
				mun.emplace_back(token);
				continue;
			}
			int index = std::stoi(token);
			tokens.push_back(index);
		}
		catch (std::exception e) {
			mun.emplace_back(token);
		}
	}
	mBatchSize = tokens[tokens[0] + 1];
	calcMat.setDropRate(tokens[tokens[0] + 2]);

	calcMat.setActName(mun[0]);
	calcMat.setOptimizer(mun[1]);
	optimizerName = mun[1];
	calcMat.setLearningRate(std::stod(mun[mun.size() - 1]));

	datas.emplace_back(std::to_string(epoch));
	datas.emplace_back(std::to_string(duration));
	datas.emplace_back(std::to_string(bestAcc));

	
}
// 1 : ���� ���̾�, 2 : ���̾� ������ 1�� ���� ����, 3 epoch
	
bool CModel::setHiddengLayerNum() {
	int numberOfHiddenLayer = tokens[0];
	/*bool isDone = inOut.output(&numOfHiddenForOuput);

	if (!isDone) {
		return false;
	}

	isDone = inOut.inputInt(&numberOfHiddenLayer);

	if (!isDone || numberOfHiddenLayer < 0) {
		return false;
 	}*/

	mHiddenLayerNumber = (size_t)numberOfHiddenLayer;
	 
	return true;
}

bool CModel::makeHidenLayer() {
	if (!setHiddengLayerNum()) {
		return false;
	}
	bool isDone = true;
	for (int index = 0; index < mHiddenLayerNumber; index++) {
		int size = tokens[index + 1];
		/*isDone = inOut.output(&sizeOfHiddenForOuput);

		if (!isDone) {
			break;
		}
		isDone = inOut.inputInt(&size);

		if (!isDone || size <= 0) {
			isDone = false;
			break;
		}*/

		CMat Layer(mBatchSize, size);

		mHiddenLayers.emplace_back(new CMat(&Layer));
	}
	
	return isDone;
}

bool CModel::makeWeight() {
	if (mHiddenLayers.empty()) {
		mHiddenLayers.emplace_back(new CMat(1, 1));
		mWeights.emplace_back(new CMat(mInputLayer->getCol(), mTargetMat->getCol()));
		mBiases.emplace_back(new CMat(mBatchSize, mWeights.at(0)->getCol()));

		bool isDone = replaceWeight();
		if (!isDone) {
			return false;
		}

		if (optimizerName.compare("adam")) {
			resetBias();
		}

		return true;
	}

	for (size_t index = 0; index <= mHiddenLayers.size(); index++) {
		if (index == 0) {
			mWeights.emplace_back(new CMat(mInputLayer->getCol(), mHiddenLayers.at(index)->getCol()));
		}
		if (index == mHiddenLayers.size()) {
			mWeights.emplace_back(new CMat(mHiddenLayers.at(index - 1)->getCol(),
				mTargetMat->getCol()));
		}
		else if (index > 0) {
			mWeights.emplace_back(new CMat(mHiddenLayers.at(index - 1)->getCol(),
				mHiddenLayers.at(index)->getCol()));
		}
		mBiases.emplace_back(new CMat(mBatchSize, mWeights.at(index)->getCol()));
	}

	bool isDone = replaceWeight();
	if (!isDone) {
		return false;
	}

	if (optimizerName.compare("adam")) {
		resetBias();
	}

	return true;
}

void CModel::resetBias() {
	std::vector<CMat*> isSwap;

	for (size_t index = 0; index < mBiases.size(); index++) {
		isSwap.emplace_back(nullptr);
	}
	for (CMat* mat : mBiases) {
		delete mat;
	}
	isSwap.swap(mBiases);
}

bool CModel::replaceWeight() {
	for (size_t index = 0; index < mWeights.size(); index++) {
		mWeights.at(index)->replaceHeRandom();
		mBiases.at(index)->replaceHeRandom();
	}

	return true;
}

bool CModel::setTestData(const std::string* imgFilePath,
	const std::string* targetFilePath) {
	std::string filePath = *imgFilePath;

	bool isDone = parsingFile.parseIDXFile(&filePath, mTestInput);
	if (!isDone) {
		return false;
	}

	mTestInput->normalizeMat();


	filePath = *targetFilePath;
	isDone = parsingFile.parseIDXFile(&filePath, mTestLabel);
	if (!isDone) {
		return false;
	}

	isDone = calcMat.oneHotEncoding(mTestLabel);
	if (!isDone) {
		return false;
	}

	return true;
}

bool CModel::makeModel(const std::string* imgFilePath,
	const std::string* targetFilePath,
	bool isLoad,
	bool isMiniBatch) {
	clearLayer();

	//mBatchSize = BATCH_SIZE_ONE;

	//if (isMiniBatch) {
	//	mBatchSize = BATCH_SIZE_FIVE_HUNDRED;
	//}
	
	std::string filePath = *imgFilePath;
	bool isDone = parsingFile.parseIDXFile(&filePath, mInputLayer);
	if (!isDone) {
		return false;
	}

	mInputLayer->normalizeMat();

	filePath = *targetFilePath;
	isDone = parsingFile.parseIDXFile(&filePath, mTargetMat);
	if (!isDone) {
		return false;
	}

	isDone = calcMat.oneHotEncoding(mTargetMat);
	if (!isDone) {
		return false;
	}

	if (isLoad) {
		isDone = loadModel();
		if (!isDone) {
			return false;
		}
		return true;
	}

	isDone = makeHidenLayer();
	if (!isDone) {
		return false;
	}

	isDone = makeWeight();
	if (!isDone) {
		return false;
	}

	return true;
}

bool CModel::saveModel() {
	size_t weightSize = mWeights.size();
	size_t biasesSize = mBiases.size();

	bool isDone = true;

	bool isExist = pathCecker.checkDir(weightDirPath, true);
	if (!isExist) {
		return false;
	}

	isExist = pathCecker.checkDir(biasesDirPath, true);
	if (!isExist) {
		return false;
	}

	for (size_t num = 0; num < weightSize; num++) {
		isExist = pathCecker.checkDir(weightDirPath);
		if (!isExist) {
			return false;
		}

		if (mWeights.at(num) == nullptr) {
			continue;
		}

		std::string fileName = "weight" + std::to_string(num) + ".bin";
		std::string filePath = weightDirPath + "/" + fileName;

		isDone = parsingFile.saveMat(mWeights.at(num), &filePath);
		if (!isDone) {
			break;
		}
	}

	if (!isDone) {
		return false;
	}

	isExist = pathCecker.checkDir(biasesDirPath);
	if (!isExist) {
		return false;
	}

	for (size_t num = 0; num < weightSize; num++) {
		if (mBiases.at(num) == nullptr) {
			continue;
		}

		std::string fileName = "bias" + std::to_string(num) + ".bin";
		std::string filePath = biasesDirPath + "/" + fileName;

		isDone = parsingFile.saveMat(mBiases.at(num), &filePath);
		if (!isDone) {
			break;
		}
	}

	return isDone;
}

bool CModel::saveBestModel() {
	size_t weightSize = mWeights.size();
	size_t biasesSize = mBiases.size();

	bool isDone = true;

	bool isExist = pathCecker.checkDir(weightDirPath + " " + rinl);
	if (!isExist) {
		return false;
	}
	
	isExist = pathCecker.checkDir(weightDirPath + " " + rinl + "/" + "weights " + std::to_string(bestAcc));
	if (!isExist) {
		return false;
	}

	isExist = pathCecker.checkDir(biasesDirPath + " " + rinl);
	if (!isExist) {
		return false;
	}

	isExist = pathCecker.checkDir(biasesDirPath + " " + rinl + "/" + "biases " + std::to_string(bestAcc));
	if (!isExist) {
		return false;
	}

	for (size_t num = 0; num < weightSize; num++) {
		if (mWeights.at(num) == nullptr) {
			continue;
		}

		std::string fileName = "weight" + std::to_string(num) + ".bin";
		std::string filePath = weightDirPath + " " + rinl + "/" + "weights " + std::to_string(bestAcc) + "/" + fileName;

		isDone = parsingFile.saveMat(mWeights.at(num), &filePath);
		if (!isDone) {
			break;
		}
	}

	if (!isDone) {
		return false;
	}

	for (size_t num = 0; num < weightSize; num++) {
		if (mBiases.at(num) == nullptr) {
			continue;
		}

		std::string fileName = "bias" + std::to_string(num) + ".bin";
		std::string filePath = biasesDirPath + " " + rinl + "/" + "biases " + std::to_string(bestAcc) + "/" + fileName;

		isDone = parsingFile.saveMat(mBiases.at(num), &filePath);
		if (!isDone) {
			break;
		}
	}

	std::string fileName = rinl + ".txt";
	std::string filePath = bestDirPath + "/" + fileName;

	datas[datas.size() - 2] = std::to_string(duration);

	datas[datas.size() - 1] = std::to_string(bestAcc);

	parsingFile.saveTxt(&datas, &filePath);

	return isDone;
}

bool CModel::loadModel() {
	int num = 0;
	std::string fileName;
	std::string filePath;;

	bool isExistFile = true;
	CMat mat;

	while (true) {
		isExistFile = pathCecker.checkDir(weightDirPath);
		if (!isExistFile) {
			break;
		}

		fileName = "weight" + std::to_string(num) + ".bin";
		filePath = weightDirPath + "/" + fileName;

		isExistFile = pathCecker.check(filePath);
		if (!isExistFile) {
			break;
		}

		isExistFile = parsingFile.parseIDXFile(&filePath, &mat);
		if (!isExistFile) {
			break;
		}

		mWeights.emplace_back(new CMat(&mat));
		if (mat.getCol() != mTargetMat->getCol()) {
			mHiddenLayers.emplace_back(new CMat(1, mat.getCol()));
		}
		isExistFile = pathCecker.checkDir(biasesDirPath);
		if (!isExistFile) {
			break;	
		}

		fileName = "bias" + std::to_string(num) + ".bin";
		filePath = biasesDirPath + "/" + fileName;

		isExistFile = pathCecker.check(filePath);
		if (!isExistFile) {
			break;
		}

		isExistFile = parsingFile.parseIDXFile(&filePath, &mat);
		if (!isExistFile) {
			break;
		}

		mBatchSize = mat.getRow();

		mBiases.emplace_back(new CMat(&mat));

		num++;
	}

	if (mWeights.size() == 0 || mBiases.size() == 0) {
		return false;
	}
	else {
		for (size_t index = 0; index < mWeights.size(); index++) {
			mHiddenLayers.emplace_back(new CMat(1, 1));
		}
		return true;
	}

	return isExistFile;
}

bool CModel::makeMiniBatch() {
	size_t batchSize = mInputLayer->getRow() / mBatchSize;
	size_t stepCol = mInputLayer->getCol();
	size_t targetCol = mTargetMat->getCol();

	for (size_t index = 0; index < batchSize; index++) {
		CMat step(mBatchSize, stepCol);
		CMat targetStep(mBatchSize, targetCol);
		for (size_t row = index * mBatchSize; row < (index + 1) * mBatchSize; row++) {
			size_t stepRow = row % mBatchSize;
			for (size_t col = 0; col < stepCol; col++) {
				step.changeVal(stepRow, col, mInputLayer->getVal(row, col));
			}
			for (size_t col = 0; col < targetCol; col++) {
				targetStep.changeVal(stepRow, col, mTargetMat->getVal(row, col));
			}
		}
		mBatch.emplace_back(new CMat(&step));
		mTargetBatch.emplace_back(new CMat(&targetStep));
	}

	if (mBatch.size() != batchSize || mTargetBatch.size() != batchSize) {
		return false;
	}

	return true;
}

bool CModel::makeTestBatch() {
	size_t batchSize = mTestInput->getRow() / mBatchSize;
	size_t stepCol = mTestInput->getCol();
	size_t targetCol = mTestLabel->getCol();

	for (size_t index = 0; index < batchSize; index++) {
		CMat step(mBatchSize, stepCol);
		CMat targetStep(mBatchSize, targetCol);
		for (size_t row = index * mBatchSize; row < (index + 1) * mBatchSize; row++) {
			size_t stepRow = row % mBatchSize;
			for (size_t col = 0; col < stepCol; col++) {
				step.changeVal(stepRow, col, mTestInput->getVal(row, col));
			}
			for (size_t col = 0; col < targetCol; col++) {
				targetStep.changeVal(stepRow, col, mTestLabel->getVal(row, col));
			}
		}
		mTestBatch.emplace_back(new CMat(&step));
		mTestTargetBatch.emplace_back(new CMat(&targetStep));
	}

	if (mTestBatch.size() != batchSize || mTestTargetBatch.size() != batchSize) {
		return false;
	}

	return true;
}

void CModel::clearLayer() {
	mInputLayer->resize(0, 0);
	mOutputLayer->resize(0, 0);
	mTargetMat->resize(0, 0);

	if (!mWeights.empty() && mWeights.size() == mBiases.size()) {
		for (int index = 0; index < mWeights.size(); index++) {
			delete mWeights.at(index);
			delete mBiases.at(index);
		}
		mWeights.clear();
		mBiases.clear();
		std::vector<CMat*>().swap(mWeights);
		std::vector<CMat*>().swap(mBiases);
	}

	if (!mHiddenLayers.empty()) {
		for (int index = 0; index < mHiddenLayers.size(); index++) {
			delete mHiddenLayers.at(index);
		}
		mHiddenLayers.clear();
		std::vector<CMat*>().swap(mHiddenLayers);
	}

	if (!mBatch.empty()) {
		for (int index = 0; index < mBatch.size(); index++) {
			delete mBatch.at(index);
			delete mTargetBatch.at(index);
		}
		mBatch.clear();
		mTargetBatch.clear();
		std::vector<CMat*>().swap(mBatch);
		std::vector<CMat*>().swap(mTargetBatch);
	}

	if (!mSplitBatch.empty()) {
		for (int row = 0; row < mSplitBatch.size(); row++) {
			for (int col = 0; col < mSplitBatch[row].size(); col++) {
				delete mSplitBatch[row].at(col);
			}
		}
		mSplitBatch.clear();
		std::vector<std::vector<CMat*>>().swap(mSplitBatch);
	}
}

bool CModel::batchLearning(size_t epoch) {
	double loss = 0;
	double tLoss = 0;
	double totalLoss = 0;
	double accuracy = 0;
	int miniEpoch = 0;
	int maxME = 0;

	bool isDone = true;

	CMat outputPerEpoch(0, 0);
	
	clock_t start, end;
	start = clock();
	for (int index = 0; index < mBatch.size(); index++) {
		*mOutputLayer = calcMat.forwardPropagation(mBatch[index],
			&mHiddenLayers, &mWeights, &mBiases, true);

		//loss += calcMat.meanSquareError(mTargetBatch[index], mOutpu tLayer);

		loss = calcMat.categoricalCrossEntropy(mTargetBatch[index], mOutputLayer);

		isDone = calcMat.backwardPropagation(epoch + 1,
			mBatch[index], mOutputLayer, mTargetBatch[index],
			&mWeights, &mHiddenLayers, &mBiases);
		if (!isDone) {
			break;
		}

		if ((loss < 0.1) || miniEpoch == 1) {

			if (loss <= 0.0) {
				exit(1);
			}

			tLoss += loss;

			if (maxME < miniEpoch) {
				maxME = miniEpoch;
			}
			miniEpoch = 0;
		}
		else {
			miniEpoch++;
			index--;
		}

		/*isDone = outputPerEpoch.pushBack(mOutputLayer);
		if (!isDone) {
			return false;
		}*/

		
	}

	if (!isDone) {
		return false;
	}

	end = clock();

	duration = (double)(end - start) / CLOCKS_PER_SEC;
	
	totalLoss = calculateTotalLoss(tLoss);
	/*accuracy = calcMat.calculateAccuracy(mTargetMat, &outputPerEpoch);*/

	std::cout << "max mini epoch is " << maxME << "\n" << "loss : " << totalLoss << "\n";

	//std::cout << "\ntraining : ";
	//isDone = inOut.printLearnStat(epoch, totalLoss, 1, 1);
	
	return isDone;
}

bool CModel::traing() {
	epoch = tokens[tokens.size() - 1];

	bool isDone = true;

	//isDone = inOut.output(&numOfEpoch);
	//if (!isDone) {
	//	return false;
	//}

	//isDone = inOut.inputInt(&epoch);
	//if (!isDone) {
	//	return false;
	//}

	isDone = makeMiniBatch();
	if (!isDone) {
		return false;
	}

	isDone = makeTestBatch();
	if (!isDone) {
		return false;
	}

	std::cout << "token clear \nbatch clear\n";

	for (size_t e = 0; e < epoch; e++) {
		//std::random_shuffle(mBatch.begin(), mBatch.end());
		datas[datas.size() - 3] = std::to_string(e + 1);
		isDone = batchLearning(e);
		if (!isDone) {
			break;
		}
		/*isDone = mtest(e);
		if (!isDone) {
			break;
		}*/
		isDone = saveModel();
		if (!isDone) {
			break;
		}
		runAll(getBatchSize(), e);
		if (bestSaveOn) {
			
			isDone = saveBestModel();
			if (!isDone) {
				break;
			}
			bestSaveOn = false;
		}
	}

	bestAcc = 0.0;

	return isDone;
}
//����
bool CModel::splitOfBatch() {
	int ySize = mBatch.size() / threadNum;

	mSplitBatch.resize(threadNum);

	int checkNum = 0;

	for (int x = 0; x < mSplitBatch.size(); x++) {
		int start = x * ySize;
		int end = (x + 1) * ySize;
		for (int index = start; index < end; index++) {
			mSplitBatch[x].emplace_back(new CMat(mBatch[index]));
			checkNum++;
		}
	}

	if (checkNum != mBatch.size()) {
		return false;
	}

	return true;
}
// ���� ����
bool CModel::propgate(double* loss, CMat* mat, size_t* epoch, int indexNum) {
	bool isDone = true;
	double threadLoss = 0.0;

	CMat threadOuput(0, 0);

	CMat threadMat(0, 0);

	std::vector<CMat*> threadHiddenLayers;

	for (int index = 0; index < mHiddenLayers.size(); index++) {
		threadHiddenLayers.emplace_back(new CMat(mHiddenLayers[index]));
	}

	for (int index = 0; index < mSplitBatch[indexNum].size(); index++) {
		int targetBatchIndex = (indexNum * mSplitBatch[indexNum].size()) + index;

 		threadOuput = calcMat.forwardPropagation(mSplitBatch[indexNum][index],
			&threadHiddenLayers, &mWeights, &mBiases);

		isDone = calcMat.backwardPropagation(*epoch,
			mSplitBatch[indexNum][index], &threadOuput, mTargetBatch[targetBatchIndex],
			&mWeights, &threadHiddenLayers, &mBiases);
		if (!isDone) {
			break;
		}

		threadLoss += calcMat.categoricalCrossEntropy(mTargetBatch[targetBatchIndex], &threadOuput);

		isDone = threadMat.pushBack(&threadOuput);
		if (!isDone) {
			return false;
		}
	}

	if (!isDone) {
		return false;
	}

	*loss = threadLoss;

	*mat = new CMat(&threadMat);

	for (int index = 0; index < threadHiddenLayers.size(); index++) {
		delete threadHiddenLayers.at(index);
	}

	return true;
}

bool CModel::threadBatchLearning(size_t epoch) {
	

	return true;
}

bool CModel::traingUseThread() {
	return true;
}

bool CModel::test(size_t saveBatchSize) {

	double loss = 0;
	double totalLoss = 0;
	double accuracy = 0;


	bool isDone = true;
	if (saveBatchSize != 0) {
		mBatchSize = saveBatchSize;
		isDone = makeMiniBatch();
		if (!isDone) {
			std::cout << "error";
		}
		isDone = makeTestBatch();
		if (!isDone) {
			std::cout << "error";
		}
	}

	int batchSize = mBatch.size();
	int testBatchSize = mTestBatch.size();

	CMat outputPerEpoch(0, 0);

	/*for (int index = 0; index < batchSize; index++) {


		*mOutputLayer = calcMat.forwardPropagation(mBatch[index],
			&mHiddenLayers, &mWeights, &mBiases);

		loss += calcMat.categoricalCrossEntropy(mTargetBatch[index], mOutputLayer);

		outputPerEpoch.pushBack(mOutputLayer);
	}

	{
		double firstAccuracy = calcMat.calculateAccuracy(mTargetMat, &outputPerEpoch);
		accuracy = firstAccuracy * batchSize;
	}

	outputPerEpoch.resize(0, 0);*/

	for (int index = 0; index < testBatchSize; index++) {
		*mOutputLayer = calcMat.testForward(mTestBatch[index],
			&mHiddenLayers, &mWeights, &mBiases);

		loss += calcMat.categoricalCrossEntropy(mTestTargetBatch[index], mOutputLayer);

		outputPerEpoch.pushBack(mOutputLayer);
	}

	{
		double secondAccuracy = calcMat.calculateAccuracy(mTestLabel, &outputPerEpoch);
		accuracy = secondAccuracy * testBatchSize;
	}

	int totalBatchSize =/* batchSize +*/ testBatchSize;

	totalLoss = loss / totalBatchSize;
	accuracy = accuracy / totalBatchSize;

	std::cout << "final test use all data \n";

	isDone = inOut.printLearnStat(0, totalLoss, accuracy, totalBatchSize);

	return isDone;
}

bool CModel::mtest(size_t saveBatchSize) {
	bool isDone = true;

	double loss = 0;
	double totalLoss = 0;
	double accuracy = 0;
	int miniEpoch = 0;
	double tLoss = 0;
	int maxME = 0;

	CMat outputPerEpoch(0, 0);

	for (int index = 0; index < mTestBatch.size(); index++) {
		*mOutputLayer = calcMat.testForward(mTestBatch[index],
			&mHiddenLayers, &mWeights, &mBiases);

		loss = calcMat.categoricalCrossEntropy(mTestTargetBatch[index], mOutputLayer);

		isDone = calcMat.backwardPropagation(0,
			mTestBatch[index], mOutputLayer, mTestTargetBatch[index],
			&mWeights, &mHiddenLayers, &mBiases);
		if (!isDone) {
			break;
		}

		if ((loss > 0.0 && loss < 0.1) || miniEpoch == 128) {
			isDone = outputPerEpoch.pushBack(mOutputLayer);
			if (!isDone) {
				return false;
			}

			if (loss <= 0.0) {
				exit(1);
			}

			tLoss += loss;

			if (maxME < miniEpoch) {
				maxME = miniEpoch;
			}
			miniEpoch = 0;
		}
		else {
			miniEpoch++;
			index--;
		}
	}
	totalLoss = tLoss / (mTestInput->getRow() / mBatchSize);
	accuracy = calcMat.calculateAccuracy(mTestLabel, &outputPerEpoch);

	std::cout << "max mini epoch is " << maxME;

	std::cout << "\ntest set : ";

	isDone = inOut.printLearnStat(saveBatchSize, totalLoss, accuracy, mTestInput->getRow());

	return isDone;
}

void CModel::runAll(size_t saveBatchSize, int epoch) {
	double loss = 0;
	double totalLoss = 0;
	double accuracy = 0;


	bool isDone = true;

	int batchSize = mBatch.size() * saveBatchSize;
	int testBatchSize = mTestBatch.size();

	CMat outputPerEpoch(0, 0);

	/*for (int index = 0; index < batchSize; index++) {


		*mOutputLayer = calcMat.forwardPropagation(mBatch[index],
			&mHiddenLayers, &mWeights, &mBiases);

		loss += calcMat.categoricalCrossEntropy(mTargetBatch[index], mOutputLayer);

		outputPerEpoch.pushBack(mOutputLayer);
	}

	{
		double firstAccuracy = calcMat.calculateAccuracy(mTargetMat, &outputPerEpoch);
		accuracy = firstAccuracy * batchSize;
	}

	outputPerEpoch.resize(0, 0);*/

	for (int index = 0; index < testBatchSize; index++) {
		*mOutputLayer = calcMat.testForward(mTestBatch[index],
			&mHiddenLayers, &mWeights, &mBiases);

		loss += calcMat.categoricalCrossEntropy(mTestTargetBatch[index], mOutputLayer);

		outputPerEpoch.pushBack(mOutputLayer);
	}

	{
		double secondAccuracy = calcMat.calculateAccuracy(mTestLabel, &outputPerEpoch);
		accuracy = secondAccuracy * testBatchSize * saveBatchSize;
	}

	int totalBatchSize = /*batchSize +*/ testBatchSize * saveBatchSize;

	if (loss <= 0.0) {
		exit(1);
	}

	totalLoss = loss / totalBatchSize;
	accuracy = accuracy / totalBatchSize;

	std::cout << "final test use all data \n";

	isDone = inOut.printLearnStat(epoch, totalLoss, accuracy, totalBatchSize); 

	if (accuracy > 0.65 && accuracy > bestAcc) {
		if (bestAcc != 0.0) {
			rmDir.removeDir(weightDirPath + std::to_string(bestAcc));
			rmDir.removeDir(biasesDirPath + std::to_string(bestAcc));
		}
		bestAcc = accuracy;
		bestSaveOn = true;
	}
}