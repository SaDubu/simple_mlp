#pragma once

#include <string>
#include <sys/stat.h>
#include <iostream>

class CPathChecker {
public:
	bool check(std::string filePath) { 
		std::cout << filePath << "\n";
		struct stat buffer;
		return (stat(filePath.c_str(), &buffer) == 0);
	}

	bool checkDir(const std::string dirPath, bool isRemove = false);

	~CPathChecker() {};
};