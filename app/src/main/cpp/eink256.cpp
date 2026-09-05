#include <jni.h>
#include <android/bitmap.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <android/log.h>

// 辅助宏：将值限制在 0-255 之间
#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

// --- 像素访问辅助类 ---

// 针对 ARGB_8888 (32位) 格式的处理
struct Pixel8888 {
    static int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        return (77 * r + 150 * g + 29 * b) >> 8;
    }

    static uint32_t pack(uint32_t original, int grayVal) {
        uint32_t alpha = original & 0xFF000000;
        return alpha | (grayVal << 16) | (grayVal << 8) | grayVal;
    }
};

// 针对 RGB_565 (16位) 格式的处理
struct Pixel565 {
    static int getGray(uint16_t color) {
        int r5 = (color >> 11) & 0x1F;
        int g6 = (color >> 5) & 0x3F;
        int b5 = color & 0x1F;

        int r8 = (r5 << 3) | (r5 >> 2);
        int g8 = (g6 << 2) | (g6 >> 4);
        int b8 = (b5 << 3) | (b5 >> 2);

        return (77 * r8 + 150 * g8 + 29 * b8) >> 8;
    }

    static uint16_t pack(uint16_t original, int grayVal) {
        int r5 = grayVal >> 3;
        int g6 = grayVal >> 2;
        int b5 = grayVal >> 3;
        return (r5 << 11) | (g6 << 5) | b5;
    }
};

// --- 模板化抖动算法 (优化暗部细节与性能) ---
template <typename T, typename Handler>
void applyDitherTemplate(void* pixelsRaw, int width, int height) {
    T* pixels = (T*)pixelsRaw;
    const int STEP = 17; // 255 / 15 = 17

    std::vector<int> currRowErr(width, 0);
    std::vector<int> nextRowErr(width, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            T originalColor = pixels[index];
            
            // 1. 获取原始灰度 (0-255)
            int rawGray = Handler::getGray(originalColor);

            // 2. 加上扩散过来的误差，并进行 Clamp 限制
            // 关键改动 A：防止暗部负误差过度累积死锁，保留暗部微弱的阶梯细节
            int gray = CLAMP(rawGray + currRowErr[x]);

            // 3. 量化 (定点数四舍五入，替换掉极其缓慢且丢失暗部阶梯的 std::round)
            // 关键改动 B：(gray + 8) / 17 能够平滑处理 0~8 范围内的极暗像素
            int level = (gray + 8) / STEP;
            if (level > 15) level = 15; // 边界保护
            int newGray = level * STEP;

            // 4. 计算量化误差
            int quantError = gray - newGray;

            // 5. 扩散误差 (Floyd-Steinberg 算法)
            if (x + 1 < width) {
                currRowErr[x + 1] += quantError * 7 / 16;
            }
            
            if (y + 1 < height) {
                if (x - 1 >= 0) nextRowErr[x - 1] += quantError * 3 / 16;
                nextRowErr[x] += quantError * 5 / 16;
                if (x + 1 < width) nextRowErr[x + 1] += quantError * 1 / 16;
            }

            // 6. 写回内存
            pixels[index] = Handler::pack(originalColor, newGray);
        }

        // 行结束：交换缓冲区
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

    __android_log_print(ANDROID_LOG_INFO, "zyymeEink256", "Dithering Bitmap: W=%d H=%d Format=%d", info.width, info.height, info.format);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyDitherTemplate<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyDitherTemplate<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
