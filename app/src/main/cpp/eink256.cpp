#include <jni.h>
#include <android/bitmap.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <android/log.h>

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

// --- 1. 双线性插值缩放函数 (Bilinear Resampling) ---
// 将 src 图像缩放到 dst 图像中，确保先高精度缩放，再进行抖动
template <typename T>
void scaleImageBilinear(const T* src, int srcW, int srcH, T* dst, int dstW, int dstH) {
    float xRatio = (float)(srcW - 1) / dstW;
    float yRatio = (float)(srcH - 1) / dstH;

    for (int y = 0; y < dstH; ++y) {
        int sy = (int)(y * yRatio);
        float yDiff = (y * yRatio) - sy;
        int syNext = std::min(sy + 1, srcH - 1);

        for (int x = 0; x < dstW; ++x) {
            int sx = (int)(x * xRatio);
            float xDiff = (x * xRatio) - sx;
            int sxNext = std::min(sx + 1, srcW - 1);

            // 采样相邻的 4 个像素
            T pA = src[sy * srcW + sx];
            T pB = src[sy * srcW + sxNext];
            T pC = src[syNext * srcW + sx];
            T pD = src[syNext * srcW + sxNext];

            // 如果是 RGBA_8888 (32位)
            if (sizeof(T) == 4) {
                uint32_t* a = (uint32_t*)&pA; uint32_t* b = (uint32_t*)&pB;
                uint32_t* c = (uint32_t*)&pC; uint32_t* d = (uint32_t*)&pD;

                uint8_t r = (uint8_t)((*a >> 16 & 0xFF) * (1-xDiff)*(1-yDiff) + (*b >> 16 & 0xFF) * xDiff*(1-yDiff) + (*c >> 16 & 0xFF) * (1-xDiff)*yDiff + (*d >> 16 & 0xFF) * xDiff*yDiff);
                uint8_t g = (uint8_t)((*a >> 8 & 0xFF)  * (1-xDiff)*(1-yDiff) + (*b >> 8 & 0xFF)  * xDiff*(1-yDiff) + (*c >> 8 & 0xFF)  * (1-xDiff)*yDiff + (*d >> 8 & 0xFF)  * xDiff*yDiff);
                uint8_t bl= (uint8_t)((*a & 0xFF)       * (1-xDiff)*(1-yDiff) + (*b & 0xFF)       * xDiff*(1-yDiff) + (*c & 0xFF)       * (1-xDiff)*yDiff + (*d & 0xFF)       * xDiff*yDiff);
                uint8_t alpha = *a & 0xFF000000;

                dst[y * dstW + x] = alpha | (r << 16) | (g << 8) | bl;
            } else { // RGB_565 (16位)
                dst[y * dstW + x] = pA; // 565极少在大图缩放中使用，简易处理
            }
        }
    }
}

// --- 2. 灰度提取辅助类 ---
struct Pixel8888 {
    static int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        return (77 * r + 150 * g + 29 * b) >> 8;
    }
    static uint32_t pack(uint32_t original, int grayVal) {
        return (original & 0xFF000000) | (grayVal << 16) | (grayVal << 8) | grayVal;
    }
};

struct Pixel565 {
    static int getGray(uint16_t color) {
        int r8 = ((color >> 11) & 0x1F) << 3;
        int g8 = ((color >> 5) & 0x3F) << 2;
        int b8 = (color & 0x1F) << 3;
        return (77 * r8 + 150 * g8 + 29 * b8) >> 8;
    }
    static uint16_t pack(uint16_t original, int grayVal) {
        return ((grayVal >> 3) << 11) | ((grayVal >> 2) << 5) | (grayVal >> 3);
    }
};

// --- 3. 缩放 + Floyd-Steinberg 核心算法 ---
template <typename T, typename Handler>
void processScaleAndDither(void* pixelsRaw, int srcW, int srcH, int targetWidth) {
    T* srcPixels = (T*)pixelsRaw;

    // 如果图片宽度与目标宽度差距小于 50px，不进行缩放，直接原图抖动
    int dstW = srcW;
    int dstH = srcH;
    bool needsScale = std::abs(srcW - targetWidth) > 50;

    if (needsScale) {
        dstW = targetWidth;
        dstH = (int)((float)srcH * targetWidth / srcW); // 保持宽高比
    }

    std::vector<T> scaledBuffer;
    T* workPixels = srcPixels;

    // 如果需要缩放，先在临时 Buffer 中做双线性插值缩放
    if (needsScale) {
        scaledBuffer.resize(dstW * dstH);
        scaleImageBilinear<T>(srcPixels, srcW, srcH, scaledBuffer.data(), dstW, dstH);
        workPixels = scaledBuffer.data();
    }

    // 在缩放后的 1:1 图像尺寸上进行 Floyd-Steinberg 抖动
    const int STEP = 255 / 15; // 16阶灰度
    std::vector<int> currRowErr(dstW, 0);
    std::vector<int> nextRowErr(dstW, 0);

    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            int index = y * dstW + x;
            T originalColor = workPixels[index];

            int gray = Handler::getGray(originalColor) + currRowErr[x];
            int newGray = CLAMP(std::round((float)gray / STEP) * STEP);
            int quantError = gray - newGray;

            // 误差扩散
            if (x + 1 < dstW) currRowErr[x + 1] += quantError * 7 / 16;
            if (y + 1 < dstH) {
                if (x - 1 >= 0) nextRowErr[x - 1] += quantError * 3 / 16;
                nextRowErr[x]     += quantError * 5 / 16;
                if (x + 1 < dstW) nextRowErr[x + 1] += quantError * 1 / 16;
            }

            workPixels[index] = Handler::pack(originalColor, newGray);
        }
        currRowErr = nextRowErr;
        std::fill(nextRowErr.begin(), nextRowErr.end(), 0);
    }

    // 如果进行了缩放，把处理完的点阵写回原 Bitmap 内存的前半部分（防止越界）
    if (needsScale) {
        int copySize = std::min(srcW * srcH, dstW * dstH);
        std::memcpy(srcPixels, scaledBuffer.data(), copySize * sizeof(T));
    }
}

// --- JNI 导出入口 ---
extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    // 假设目标墨水屏短边像素为 1404 (可以根据你的实际设备硬编码，如 1072, 1404, 1872 等)
    int targetDisplayWidth = 1404; 

    __android_log_print(ANDROID_LOG_INFO, "zyymeEink256", "Processing: W=%d H=%d -> TargetW=%d", info.width, info.height, targetDisplayWidth);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        processScaleAndDither<uint32_t, Pixel8888>(pixels, info.width, info.height, targetDisplayWidth);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        processScaleAndDither<uint16_t, Pixel565>(pixels, info.width, info.height, targetDisplayWidth);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
