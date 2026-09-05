#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cstring>
#include <android/log.h>

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

// 针对 ARGB_8888 格式
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

// 针对 RGB_565 格式
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

// 预算 16 阶灰度查找表 (LUT) 以替代 round() 和除法
static int LUT_INITIALIZED = 0;
static uint8_t GRAY_LUT[768]; // 支持 [-255, 512] 的灰度范围索引

static void initLUT() {
    if (LUT_INITIALIZED) return;
    const float STEP = 255.0f / 15.0f; // 16阶灰度步长 17
    for (int i = -255; i <= 512; ++i) {
        int clamped = CLAMP(i);
        int quantized = (int)(std::round(clamped / STEP) * STEP);
        GRAY_LUT[i + 255] = (uint8_t)CLAMP(quantized);
    }
    LUT_INITIALIZED = 1;
}

template <typename T, typename Handler>
void applyDitherFast(void* pixelsRaw, int width, int height) {
    initLUT();
    T* pixels = (T*)pixelsRaw;

    // 使用原生数组 & 指针交换，杜绝 std::vector 深拷贝
    int* errBuf0 = new int[width + 2](); // 左右各留 1 个 padding，免去越界判断
    int* errBuf1 = new int[width + 2]();

    int* currRowErr = errBuf0 + 1; // 偏移 1 个单位，允许 [-1] 访问
    int* nextRowErr = errBuf1 + 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            T originalColor = pixels[x];

            // 1. 获取灰度 & 加误差
            int gray = Handler::getGray(originalColor) + currRowErr[x];

            // 2. 极速 O(1) 查表量化
            int newGray = GRAY_LUT[gray + 255];

            // 3. 计算量化误差
            int quantError = gray - newGray;

            // 4. 移位扩散误差 (全整数位移，免除法)
            // 向右: 7/16
            currRowErr[x + 1] += (quantError * 7) >> 4;

            if (y + 1 < height) {
                // 左下: 3/16, 下: 5/16, 右下: 1/16
                nextRowErr[x - 1] += (quantError * 3) >> 4;
                nextRowErr[x]     += (quantError * 5) >> 4;
                nextRowErr[x + 1] += quantError >> 4;
            }

            // 5. 写回内存
            pixels[x] = Handler::pack(originalColor, newGray);
        }

        // 移动到下一行指针
        pixels += width;

        // 零开销交换行缓冲区
        std::swap(currRowErr, nextRowErr);
        // 清空下一行累积误差
        std::memset(nextRowErr - 1, 0, (width + 2) * sizeof(int));
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
