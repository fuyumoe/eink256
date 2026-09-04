#include <jni.h>
#include <android/bitmap.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
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
        uint32_t alpha = original & 0xFF000000;
        return alpha | (grayVal << 16) | (grayVal << 8) | grayVal;
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
        int r5 = grayVal >> 3;
        int g6 = grayVal >> 2;
        int b5 = grayVal >> 3;
        return (r5 << 11) | (g6 << 5) | b5;
    }
};

// --- 高性能 C++ 抖动 + 边缘锐化处理 ---
template <typename T, typename Handler>
void applySharpenAndDither(void* pixelsRaw, int width, int height) {
    T* pixels = (T*)pixelsRaw;
    const int STEP = 255 / (16 - 1); // 16阶灰度

    std::vector<int> currRowErr(width, 0);
    std::vector<int> nextRowErr(width, 0);

    // 一维灰度缓存，加速后续锐化与抖动计算
    std::vector<int> grayBuffer(width * height);
    for (int i = 0; i < width * height; ++i) {
        grayBuffer[i] = Handler::getGray(pixels[i]);
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            int gray = grayBuffer[index];

            // --- 步骤 A: 极轻量级 C++ 边缘锐化 (拉普拉斯高通滤波) ---
            // 只有在非边缘像素时进行 3x3 锐化计算，消除双线性插值带来的发虚感
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int center = gray;
                int top    = grayBuffer[index - width];
                int bottom = grayBuffer[index + width];
                int left   = grayBuffer[index - 1];
                int right  = grayBuffer[index + 1];

                // 经典轻度锐化算子：5 * center - (top + bottom + left + right)
                int sharpened = 5 * center - (top + bottom + left + right);
                // 混合原图与锐化效果 (0.75 原图 + 0.25 锐化)
                gray = (gray * 3 + CLAMP(sharpened)) >> 2;
            }

            // --- 步骤 B: 叠加 Floyd-Steinberg 扩散误差 ---
            gray += currRowErr[x];

            // --- 步骤 C: 16阶量化 ---
            int newGray = std::round((float)gray / STEP) * STEP;
            newGray = CLAMP(newGray);

            int quantError = gray - newGray;

            // --- 步骤 D: 误差扩散 (Floyd-Steinberg) ---
            if (x + 1 < width) {
                currRowErr[x + 1] += quantError * 7 / 16;
            }
            if (y + 1 < height) {
                if (x - 1 >= 0) nextRowErr[x - 1] += quantError * 3 / 16;
                nextRowErr[x] += quantError * 5 / 16;
                if (x + 1 < width) nextRowErr[x + 1] += quantError * 1 / 16;
            }

            // --- 步骤 E: 写回内存 ---
            pixels[index] = Handler::pack(pixels[index], newGray);
        }

        currRowErr = nextRowErr;
        std::fill(nextRowErr.begin(), nextRowErr.end(), 0);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applySharpenAndDither<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applySharpenAndDither<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
