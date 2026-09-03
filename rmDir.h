#pragma once


#include <fcntl.h>
#include <sys/stat.h>
#include <string>

class CRMDir {
public :
	bool makeDir(const std::string& path) {
		return (mkdir(path.c_str(), 0755) == 0);
	}

	bool removeDir(const std::string& path);
	~CRMDir() {};
};