#include "rmDir.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

bool CRMDir::removeDir(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        return false;  // 디렉토리 열기에 실패
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // "." 및 ".."는 무시
        if (std::string(entry->d_name) == "." || std::string(entry->d_name) == "..") {
            continue;
        }

        std::string fullPath = path + "/" + entry->d_name;
        if (entry->d_type == DT_DIR) {
            // 하위 디렉토리인 경우 재귀 호출
            if (!removeDir(fullPath)) {
                closedir(dir);
                return false;
            }
        } else {
            // 파일인 경우 삭제
            if (remove(fullPath.c_str()) != 0) {
                closedir(dir);
                return false;
            }
        }
    }

    closedir(dir);
    // 이제 빈 디렉토리를 삭제
    return rmdir(path.c_str()) == 0;
}