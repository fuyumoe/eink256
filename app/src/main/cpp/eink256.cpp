#include <jni.h>
#include <android/bitmap.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// 消除浮点数，用位移和整数运算代替
// 255 / 15 = 17
static inline int quantize16(int gray) {
    if (gray <= 0) return 0;
    if (gray >= 255) return 255;
    // (gray + 8) / 17 的快速近似：取整量化
    return ((gray + 8) / 17) * 17;
}

template <typename T, typename Handler>
void applyDitherFastFS(uint8_t* pixelsBase, int width, int height, uint32_t stride) {
    // 使用 C 原生栈/堆内存，避免 std::vector 构造开销
    int* currRowErr = (int*)calloc(width, sizeof(int));
    int* nextRowErr = (int*)calloc(width, sizeof(int));

    for (int y = 0; y < height; ++y) {
        T* line = (T*)(pixelsBase + y * stride);

        for (int x = 0; x < width; ++x) {
            T originalColor = line[x];

            int gray = Handler::getGray(originalColor) + currRowErr[x];

            // 替代 std::round((float)gray / STEP) * STEP
            int newGray = quantize16(gray);

            int quantError = gray - newGray;

            // 位运算与位移加速误差扩散 (*7/16, *3/16, *5/16, *1/16)
            // 向右 (7/16)
            if (x + 1 < width) {
                currRowErr[x + 1] += (quantError * 7) >> 4;
            }

            // 向下一行
            if (y + 1 < height) {
                if (x > 0) nextRowErr[x - 1] += (quantError * 3) >> 4;
                nextRowErr[x] += (quantError * 5) >> 4;
                if (x + 1 < width) nextRowErr[x + 1] += quantError >> 4;
            }

            line[x] = Handler::pack(originalColor, newGray);
        }

        // 快速指针交换，无需内存拷贝
        int* temp = currRowErr;
        currRowErr = nextRowErr;
        nextRowErr = temp;
        memset(nextRowErr, 0, width * sizeof(int));
    }

    free(currRowErr);
    free(nextRowErr);
}
