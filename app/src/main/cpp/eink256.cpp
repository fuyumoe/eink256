#include <jni.h>
#include <android/bitmap.h>
#include <vector>
#include <cstring>

// --- 针对 ARGB_8888 (32位) ---
struct Pixel8888 {
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        // 严格对齐 koplugin: 0.299R + 0.587G + 0.114B (定点数优化: 77R + 150G + 29B >> 8)
        return (77 * r + 150 * g + 29 * b) >> 8;
    }

    static inline uint32_t pack(uint32_t original, int grayVal) {
        uint32_t alpha = original & 0xFF000000;
        return alpha | (grayVal << 16) | (grayVal << 8) | grayVal;
    }
};

// --- 针对 RGB_565 (16位) ---
struct Pixel565 {
    static inline int getGray(uint16_t color) {
        int r5 = (color >> 11) & 0x1F;
        int g6 = (color >> 5) & 0x3F;
        int b5 = color & 0x1F;

        // 对齐 koplugin 的 RGB565 扩展方式
        int r8 = (r5 << 3) | (r5 >> 2);
        int g8 = (g6 << 2) | (g6 >> 4);
        int b8 = (b5 << 3) | (b5 >> 2);

        return (77 * r8 + 150 * g8 + 29 * b8) >> 8;
    }

    static inline uint16_t pack(uint16_t original, int grayVal) {
        // 对齐 koplugin 的 RGB565 打包：g5=gray>>3, g6=gray>>2
        int g5 = grayVal >> 3;
        int g6 = grayVal >> 2;
        return (g5 << 11) | (g6 << 5) | g5;
    }
};

template <typename T, typename Handler>
void applyDitherTemplate(void* pixelsRaw, int width, int height) {
    T* pixels = (T*)pixelsRaw;
    const int STEP = 17;

    std::vector<int> currRowErrBuf(width, 0);
    std::vector<int> nextRowErrBuf(width, 0);
    int* currRowErr = currRowErrBuf.data();
    int* nextRowErr = nextRowErrBuf.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            T originalColor = pixels[index];
            
            int gray = Handler::getGray(originalColor) + currRowErr[x];

            // 1. 严格对齐 koplugin 查表量化逻辑: floor((gray + 8) / 17) * 17
            int level = (gray + 8) / 17;
            if (level < 0) level = 0;
            if (level > 15) level = 15;
            int newGray = level * STEP;

            // 2. 算误差
            int quantError = gray - newGray;

            // 3. Floyd-Steinberg 扩散 (完全一致)
            if (x + 1 < width) {
                currRowErr[x + 1] += (quantError * 7) >> 4;
            }
            
            if (y + 1 < height) {
                if (x > 0) nextRowErr[x - 1] += (quantError * 3) >> 4;
                nextRowErr[x] += (quantError * 5) >> 4;
                if (x + 1 < width) nextRowErr[x + 1] += (quantError * 1) >> 4;
            }

            pixels[index] = Handler::pack(originalColor, newGray);
        }

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
