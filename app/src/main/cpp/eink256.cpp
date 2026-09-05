#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <utility>
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
void applyDitherFast(void* pixelsRaw, int width, int height) {
    if (width <= 0 || height <= 0) return;
    initLUT();
    T* pixels = (T*)pixelsRaw;

    // 多申请几个空间防止极端情况越界
    int* errBuf0 = new int[width + 4]();
    int* errBuf1 = new int[width + 4]();

    int* currRowErr = errBuf0 + 2; 
    int* nextRowErr = errBuf1 + 2;

    int lastRow = height - 1;

    for (int y = 0; y < height; ++y) {
        bool hasNextRow = (y < lastRow);

        // 针对非最后一行的高速循环（内层无任何 if 分支！）
        if (hasNextRow) {
            for (int x = 0; x < width; ++x) {
                T originalColor = pixels[x];
                int gray = Handler::getGray(originalColor) + currRowErr[x];
                int newGray = GRAY_LUT[gray + 255];
                int quantError = gray - newGray;

                currRowErr[x + 1] += (quantError * 7) >> 4;

                nextRowErr[x - 1] += (quantError * 3) >> 4;
                nextRowErr[x]     += (quantError * 5) >> 4;
                nextRowErr[x + 1] += quantError >> 4;

                pixels[x] = Handler::pack(originalColor, newGray);
            }
        } else {
            // 最后一行：不需要向下传递误差，彻底剥离分支
            for (int x = 0; x < width; ++x) {
                T originalColor = pixels[x];
                int gray = Handler::getGray(originalColor) + currRowErr[x];
                int newGray = GRAY_LUT[gray + 255];
                int quantError = gray - newGray;

                currRowErr[x + 1] += (quantError * 7) >> 4;
                pixels[x] = Handler::pack(originalColor, newGray);
            }
        }

        pixels += width;
        
        // 交换缓冲区
        std::swap(currRowErr, nextRowErr);

        // 【优化点】：不再使用全行 memset，改为精准清除下一行即将用到的区域
        // 这样比 memset 整行快数倍
        nextRowErr[-1] = 0;
        nextRowErr[0]  = 0;
        nextRowErr[1]  = 0;
        // 顺便把尾部可能被污染的几个关键位置清零
        nextRowErr[width - 1] = 0;
        nextRowErr[width]     = 0;
        nextRowErr[width + 1] = 0;
    }

    delete[] errBuf0;
    delete[] errBuf1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyDitherFast<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyDitherFast<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
