#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <utility>
#include <omp.h> // 引入 OpenMP 头文件
#include <android/log.h>

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

struct Pixel8888 {
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        return (77 * r + 150 * g + 29 * b) >> 8;
    }

    static inline uint32_t pack(uint32_t original, int grayVal) {
        return (original & 0xFF000000) | (grayVal << 16) | (grayVal << 8) | grayVal;
    }
};

struct Pixel565 {
    static inline int getGray(uint16_t color) {
        int r5 = (color >> 11) & 0x1F;
        int g6 = (color >> 5) & 0x3F;
        int b5 = color & 0x1F;

        int r8 = (r5 << 3) | (r5 >> 2);
        int g8 = (g6 << 2) | (g6 >> 4);
        int b8 = (b5 << 3) | (b5 >> 2);

        return (77 * r8 + 150 * g8 + 29 * b8) >> 8;
    }

    static inline uint16_t pack(uint16_t original, int grayVal) {
        return ((grayVal >> 3) << 11) | ((grayVal >> 2) << 5) | (grayVal >> 3);
    }
};

static int LUT_INITIALIZED = 0;
static uint8_t GRAY_LUT[768];

static void initLUT() {
    if (LUT_INITIALIZED) return;
    for (int i = -255; i <= 512; ++i) {
        int clamped = CLAMP(i);
        int quantized = ((clamped + 8) / 17) * 17;
        GRAY_LUT[i + 255] = (uint8_t)CLAMP(quantized);
    }
    LUT_INITIALIZED = 1;
}

template <typename T, typename Handler>
void applyDitherWavefront(void* pixelsRaw, int width, int height) {
    initLUT();
    T* pixels = (T*)pixelsRaw;

    // 为全图准备误差传递矩阵（按行存储）
    // 为了防止多线程竞争，每一行分配独立的误差 Buffer
    int** errBuffers = new int*[height];
    for (int i = 0; i < height; ++i) {
        errBuffers[i] = new int[width + 2](); // 带 padding
    }

    // 计算反角斜线（Wavefront）的总步数: width + height - 1
    int numDiagonals = width + height - 1;

    // 使用 OpenMP 将对角线波浪交由多核 CPU 并行处理
    for (int diag = 0; diag < numDiagonals; ++diag) {
        // 计算当前对角线上的起始与结束行
        int startY = std::max(0, diag - width + 1);
        int endY = std::min(height - 1, diag);

        #pragma omp parallel for schedule(static)
        for (int y = startY; y <= endY; ++y) {
            int x = diag - y;

            T* rowPixels = pixels + y * width;
            T originalColor = rowPixels[x];

            int* currRowErr = errBuffers[y] + 1;
            int gray = Handler::getGray(originalColor) + currRowErr[x];
            int newGray = GRAY_LUT[gray + 255];
            int quantError = gray - newGray;

            // 1. 向右扩散 (同一行)
            currRowErr[x + 1] += (quantError * 7) >> 4;

            // 2. 向下一行扩散 (线程安全：下一行的 x-1, x, x+1 此时还没有被下个线程读取)
            if (y + 1 < height) {
                int* nextRowErr = errBuffers[y + 1] + 1;
                #pragma omp atomic
                nextRowErr[x - 1] += (quantError * 3) >> 4;
                #pragma omp atomic
                nextRowErr[x]     += (quantError * 5) >> 4;
                #pragma omp atomic
                nextRowErr[x + 1] += quantError >> 4;
            }

            rowPixels[x] = Handler::pack(originalColor, newGray);
        }
    }

    // 释放内存
    for (int i = 0; i < height; ++i) {
        delete[] errBuffers[i];
    }
    delete[] errBuffers;
}

extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyDitherWavefront<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyDitherWavefront<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
