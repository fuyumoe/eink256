#include <jni.h>
#include <android/bitmap.h>
#include <vector>
#include <algorithm>

// KOReader 风格：纯整数极限 CLAMP 宏
#define CLAMP255(v) ((v) < 0 ? 0 : ((v) > 255 ? 255 : (v)))

// 1. ARGB_8888 (32位) 极速指针流处理
static void dither_ARGB_8888(uint32_t* pixels, int width, int height) {
    // 16阶灰度步长 = 17 (255 / 15)
    const int STEP = 17;
    const int HALF_STEP = 8; // 整数四舍五入偏移量

    // 两行滑动误差缓冲区
    std::vector<int> currErr(width + 2, 0);
    std::vector<int> nextErr(width + 2, 0);

    for (int y = 0; y < height; ++y) {
        std::fill(nextErr.begin(), nextErr.end(), 0);

        for (int x = 0; x < width; ++x) {
            uint32_t color = *pixels;

            // 解包 R, G, B
            int r = (color >> 16) & 0xFF;
            int g = (color >> 8) & 0xFF;
            int b = color & 0xFF;

            // 经典 256 阶灰度转换 (77R + 150G + 29B) >> 8
            int gray = ((77 * r + 150 * g + 29 * b) >> 8) + currErr[x + 1];

            // 16 阶量化
            int clamped = CLAMP255(gray);
            int newGray = ((clamped + HALF_STEP) / STEP) * STEP;
            if (newGray > 255) newGray = 255;

            // 量化误差
            int err = gray - newGray;

            // Floyd-Steinberg 7/16, 3/16, 5/16, 1/16 扩散
            currErr[x + 2] += (err * 7) >> 4;
            nextErr[x]     += (err * 3) >> 4;
            nextErr[x + 1] += (err * 5) >> 4;
            nextErr[x + 2] += err >> 4;

            // 写回内存并推进指针
            uint32_t alpha = color & 0xFF000000;
            *pixels++ = alpha | (newGray << 16) | (newGray << 8) | newGray;
        }

        std::swap(currErr, nextErr);
    }
}

// 2. RGB_565 (16位) 极速指针流处理
static void dither_RGB_565(uint16_t* pixels, int width, int height) {
    const int STEP = 17;
    const int HALF_STEP = 8;

    std::vector<int> currErr(width + 2, 0);
    std::vector<int> nextErr(width + 2, 0);

    for (int y = 0; y < height; ++y) {
        std::fill(nextErr.begin(), nextErr.end(), 0);

        for (int x = 0; x < width; ++x) {
            uint16_t color = *pixels;

            // 快速将 565 扩展为 888 灰度
            int r8 = ((color >> 11) & 0x1F) << 3;
            int g8 = ((color >> 5) & 0x3F) << 2;
            int b8 = (color & 0x1F) << 3;

            int gray = ((77 * r8 + 150 * g8 + 29 * b8) >> 8) + currErr[x + 1];

            int clamped = CLAMP255(gray);
            int newGray = ((clamped + HALF_STEP) / STEP) * STEP;
            if (newGray > 255) newGray = 255;

            int err = gray - newGray;

            currErr[x + 2] += (err * 7) >> 4;
            nextErr[x]     += (err * 3) >> 4;
            nextErr[x + 1] += (err * 5) >> 4;
            nextErr[x + 2] += err >> 4;

            int r5 = newGray >> 3;
            int g6 = newGray >> 2;
            int b5 = newGray >> 3;

            *pixels++ = (r5 << 11) | (g6 << 5) | b5;
        }

        std::swap(currErr, nextErr);
    }
}

// JNI 函数入口
extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        dither_ARGB_8888(static_cast<uint32_t*>(pixels), info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        dither_RGB_565(static_cast<uint16_t*>(pixels), info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
