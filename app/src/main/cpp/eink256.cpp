#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <android/log.h>

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

// --- 预计算 Gamma 1.8 查表与反查表（解决暗部/发尾过渡不自然） ---
static unsigned char GAMMA_LUT[256];
static unsigned char INV_GAMMA_LUT[256];
static bool is_gamma_inited = false;

static void initGammaTable() {
    if (is_gamma_inited) return;
    for (int i = 0; i < 256; ++i) {
        float norm = i / 255.0f;
        GAMMA_LUT[i] = (unsigned char)(powf(norm, 1.8f) * 255.0f + 0.5f); // 墨水屏微调 Gamma 1.8
        INV_GAMMA_LUT[i] = (unsigned char)(powf(norm, 1.0f / 1.8f) * 255.0f + 0.5f);
    }
    is_gamma_inited = true;
}

// --- 像素访问辅助类 ---

struct Pixel8888 {
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        int linearGray = (77 * r + 150 * g + 29 * b) >> 8;
        return GAMMA_LUT[linearGray]; // 应用 Gamma 映射
    }

    static inline uint32_t pack(uint32_t original, int grayVal) {
        uint32_t alpha = original & 0xFF000000;
        int outGray = INV_GAMMA_LUT[grayVal]; // 反映射回显示空间
        return alpha | (outGray << 16) | (outGray << 8) | outGray;
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

        int linearGray = (77 * r8 + 150 * g8 + 29 * b8) >> 8;
        return GAMMA_LUT[linearGray];
    }

    static inline uint16_t pack(uint16_t original, int grayVal) {
        int outGray = INV_GAMMA_LUT[grayVal];
        int r5 = outGray >> 3;
        int g6 = outGray >> 2;
        int b5 = outGray >> 3;
        return (r5 << 11) | (g6 << 5) | b5;
    }
};

// --- 优化后的抖动算法 ---
template <typename T, typename Handler>
void applyDitherTemplate(void* pixelsRaw, int width, int height) {
    initGammaTable();
    T* pixels = (T*)pixelsRaw;
    const int STEP = 17;

    // 原生 C 指针缓冲区，提升 CPU Cache 命中率
    std::vector<int> currRowErrBuf(width, 0);
    std::vector<int> nextRowErrBuf(width, 0);
    int* currRowErr = currRowErrBuf.data();
    int* nextRowErr = nextRowErrBuf.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            T originalColor = pixels[index];
            
            // 1. 获取经过 Gamma 矫正后的灰度
            int rawGray = Handler::getGray(originalColor);

            // 2. 叠加扩散误差并约束范围
            int gray = CLAMP(rawGray + currRowErr[x]);

            // 3. 优化量化逻辑：
            // ① 使用无周期的空间 Hash 混淆，代替原来的 ((x^y)&1) 棋盘格，彻底消除了缩放时的方形干涉色块
            // ② 仅在中暗区 (gray < 180) 叠加微弱扰动打散阶梯，较亮区域保持纯净不加噪点
            int hashNoise = ((x * 1259 + y * 29573) ^ (x >> 1)) & 1; 
            int ditherNoise = (gray < 180) ? hashNoise : 0;

            int level = (gray + 8 + ditherNoise) / STEP;
            if (level < 0) level = 0;
            if (level > 15) level = 15;
            int newGray = level * STEP;

            // 4. 计算量化误差
            int quantError = gray - newGray;

            // 5. Floyd-Steinberg 扩散（位运算 >>4 代替 /16）
            if (x + 1 < width) {
                currRowErr[x + 1] += (quantError * 7) >> 4;
            }
            
            if (y + 1 < height) {
                if (x > 0) nextRowErr[x - 1] += (quantError * 3) >> 4;
                nextRowErr[x] += (quantError * 5) >> 4;
                if (x + 1 < width) nextRowErr[x + 1] += (quantError * 1) >> 4;
            }

            // 6. 转换回色彩并写回内存
            pixels[index] = Handler::pack(originalColor, newGray);
        }

        // 缓冲区快速交换
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
