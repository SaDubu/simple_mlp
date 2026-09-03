#include "cio.h"

#include <string>
#include <iostream>
#include <limits>

bool CIo::inputInt(int* value) {
	if (!(std::cin >> mInt)) {
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		return false;
	}
	
	*value = mInt;

	return true;
}

bool CIo::inputDouble(double* value) {
	if (!(std::cin >> mDouble)) {
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		return false;
	}

	*value = mDouble;

	return true;
}

bool CIo::inputString(std::string* value) {
	if (!(std::cin >> mString)) {
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		return false;
	}

	*value = mString;

	return true;
}

bool CIo::inputChar(char* value) {
	if (!(std::cin >> mChar)) {
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		return false;
	}

	*value = mChar;

	return true;
}

bool CIo::output(const std::string* text) {
	std::cout << "Please enter something related to " << *text << ".  :  ";
	return true;
}

bool CIo::printLearnStat(size_t epoch, double totalLoss, double accuracy, size_t inputLayerRow) {
	std::cout << epoch + 1 << " epoch Loss |" << totalLoss << "|, accuracy |" << accuracy << "|\n";
	std::cout << "Number of correct answers : " << accuracy * inputLayerRow << "\n";

	return true;
}