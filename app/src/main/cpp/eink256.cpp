#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <vector>
#include <cstring>

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

// --- 像素访问辅助类（严格按 dither256 的 0.299/0.587/0.114 权重） ---

struct Pixel8888 {
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        // 严格对齐 koplugin: 0.299*R + 0.587*G + 0.114*B
        return (int)(0.299f * r + 0.587f * g + 0.114f * b);
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

        return (int)(0.299f * r8 + 0.587f * g8 + 0.114f * b8);
    }

    static inline uint16_t pack(uint16_t original, int grayVal) {
        int r5 = grayVal >> 3;
        int g6 = grayVal >> 2;
        int b5 = grayVal >> 3;
        return (r5 << 11) | (g6 << 5) | b5;
    }
};

// --- 纯净 Floyd-Steinberg 算法（与 koplugin 完全一致） ---
template <typename T, typename Handler>
void applyDitherTemplate(void* pixelsRaw, int width, int height) {
    T* pixels = (T*)pixelsRaw;
    const int STEP = 17;

    // 误差缓冲区
    std::vector<int> currRowErrBuf(width, 0);
    std::vector<int> nextRowErrBuf(width, 0);
    int* currRowErr = currRowErrBuf.data();
    int* nextRowErr = nextRowErrBuf.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            T originalColor = pixels[index];
            
            // 1. 获取灰度
            int rawGray = Handler::getGray(originalColor);

            // 2. 叠加上步扩散过来的误差
            int gray = rawGray + currRowErr[x];

            // 3. 严格对齐 koplugin 的 16 阶量化逻辑：floor((gray + 8) / 17) * 17
            int level = (gray + 8) / 17;
            if (level < 0) level = 0;
            if (level > 15) level = 15;
            int newGray = level * STEP;

            // 4. 计算量化误差
            int quantError = gray - newGray;

            // 5. Floyd-Steinberg 误差扩散矩阵 (7/16, 3/16, 5/16, 1/16)
            if (x + 1 < width) {
                currRowErr[x + 1] += (quantError * 7) >> 4;
            }
            
            if (y + 1 < height) {
                if (x > 0) nextRowErr[x - 1] += (quantError * 3) >> 4;
                nextRowErr[x] += (quantError * 5) >> 4;
                if (x + 1 < width) nextRowErr[x + 1] += (quantError * 1) >> 4;
            }

            // 6. 写回内存
            pixels[index] = Handler::pack(originalColor, newGray);
        }

        // 缓冲区交换并清空下一行
        std::swap(currRowErr, nextRowErr);
        std::memset(nextRowErr, 0, width * sizeof(int));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyDitherTemplate<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyDitherTemplate<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
