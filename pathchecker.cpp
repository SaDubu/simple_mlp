#include "pathchecker.h"
#include "rmDir.h"

bool CPathChecker::checkDir(const std::string dirPath, bool isRemove) {
	bool isDone = check(dirPath);

	if (isRemove && isDone) {
		CRMDir removeDir;
		isDone = removeDir.removeDir(dirPath);
	}

	if (!isDone) {
		CRMDir makeDir;
		isDone = makeDir.makeDir(dirPath);
		return isDone;
	}

	return isDone;
}