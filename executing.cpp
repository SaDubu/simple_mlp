#include "executing.h"
#include "cio.h"
#include "calcmat.h"
#include "pathchecker.h"
#include "model.h"
#include "threadNumCheck.h"

#include <iostream>
#include <stddef.h>

CModel model;
CPathChecker pathChecker;
CThreadCheck threadChecker;
CIo input;

bool isBatch = true;
std::string chooseSave = "";

bool CExecute::actModel() {
	bool isDone = pathChecker.checkDir(saveModelPath);
	if (!isDone) {
		return false;
	}

	int chooseNum = 0;
	model.setTestData(&testImgPath, &testLabelPath);
	while (true) {
		std::string a = "";
		std::getline(std::cin, a);
		if (a.compare("no") == 0) {
			std::cout << "no\n";
			chooseNum = 4;
		}
		if (chooseNum == 0) {
			isDone = pathChecker.checkDir(saveBestPath);
			model.setS(a);

			isDone = model.makeModel(&trianImgPath, &trianLabelPath, false, isBatch);
			model.traing();

			chooseNum = 4;

			exit(1);

			/*isBatch = false;
			bool isDone = proceed();
			if (!isDone) {
				continue;
			}
			model.traing();*/
		}
		else if (chooseNum == 1) {
			
			isBatch = true;
			bool isDone = proceed();
			if (!isDone) {
				continue;
			}

			isDone = model.traing();
			if (!isDone) {
				std::cout << "ERROR!!\n";
			}
		}
		else if (chooseNum == 2) {
			isBatch = true;
			int threadNumToUse = model.getUsingThreadNum();
			/*bool isUseThraed = threadChecker.checkNum(threadNumToUse);
			if (!isUseThraed) {
				std::cout << "can't use " << threadNumToUse << " of thread";
				continue;
			}*/

			bool isDone = proceed();
			if (!isDone) {
				continue;
			}

			isDone = model.traingUseThread();
			if (!isDone) {
				std::cout << "ERROR!!\n";
			}

		}
		else if (chooseNum == 3) {
			isBatch = false;
			bool is = model.makeModel(&trianImgPath, &trianLabelPath, true, isBatch);
			if (!is) {
				std::cout << "No files were saved.\n";
				continue;
			}
			std::cout << "test start\n";

			model.test(model.getBatchSize());
			if (!isDone) {
				std::cout << "ERROR!!\n";
			}
		}
		else if (chooseNum == 4) {
			std::cout << "Exit\n";
			model.clearLayer();
			break;
		}
		else {
			std::cout << "You must enter the correct number \n";
		}
	}

	return true;
}

bool CExecute::proceed() {
	bool isDone = model.makeModel(&trianImgPath, &trianLabelPath, true, isBatch);
	if (!isDone) {
		std::cout << "No files were saved.\n";
		isDone = model.makeModel(&trianImgPath, &trianLabelPath, false, isBatch);
	}
	else {
		std::cout << "Would you like to import a saved file? (yes or no) : ";
		if (!input.inputString(&chooseSave)) {
			std::cout << "False\n";
			return false;
		}
		if (chooseSave.compare("no")) {
			isDone = model.makeModel(&trianImgPath, &trianLabelPath, false, isBatch);
		}
		else {
			std::cout << "Maybe you answered \"yes\"\n";
		}
	}
	return true;
}