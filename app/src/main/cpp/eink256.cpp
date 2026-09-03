#include <jni.h>
#include <android/bitmap.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <android/log.h>

#define LOG_TAG "Eink256Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define CLAMP_255(val) ((val) < 0 ? 0 : ((val) > 255 ? 255 : (val)))

// --- 预计算 Gamma 2.2 查找表 (LUT) ---
// 解决缺陷二：在循环内部如果调用 powf 会严重卡顿，用 static 数组做 O(1) 查表转换
static uint8_t sRGB_to_Linear[256];
static uint8_t Linear_to_sRGB[256];
static bool isGammaTableInitialized = false;

// 线程安全/一次性初始化 Gamma 查找表
static void initGammaTables() {
    if (isGammaTableInitialized) return;
    for (int i = 0; i < 256; ++i) {
        float norm = i / 255.0f;
        
        // 1. sRGB 空间转换到物理线性空间 (Gamma 2.2 解码)
        float linear = std::pow(norm, 2.2f);
        sRGB_to_Linear[i] = static_cast<uint8_t>(std::round(linear * 255.0f));
        
        // 2. 物理线性空间转回 sRGB 空间 (Gamma 2.2 编码)
        float srgb = std::pow(norm, 1.0f / 2.2f);
        Linear_to_sRGB[i] = static_cast<uint8_t>(std::round(srgb * 255.0f));
    }
    isGammaTableInitialized = true;
}

// --- 统一像素提取与 Gamma 映射 ---

// 解决缺陷一：针对 ARGB_8888 提取完整 8 位（256 阶）输入，并解包到线性空间
inline int getLinearLuminance8888(uint32_t color) {
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = color & 0xFF;

    // 1. 经典人眼感知灰度公式得到 sRGB 灰度 (0-255)
    uint8_t srgbGray = static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);

    // 2. 通过 LUT 转为线性空间灰度，防止暗部阶梯断层
    return sRGB_to_Linear[srgbGray];
}

// 解决缺陷一：针对 RGB_565，进行无损位深扩展（5/6 bit -> 8 bit）后再解包到线性空间
inline int getLinearLuminance565(uint16_t color) {
    int r5 = (color >> 11) & 0x1F;
    int g6 = (color >> 5) & 0x3F;
    int b5 = color & 0x1F;

    // 位深精确扩展到 256 阶 (0-255)
    int r8 = (r5 << 3) | (r5 >> 2);
    int g8 = (g6 << 2) | (g6 >> 4);
    int b8 = (b5 << 3) | (b5 >> 2);

    uint8_t srgbGray = static_cast<uint8_t>((77 * r8 + 150 * g8 + 29 * b8) >> 8);
    return sRGB_to_Linear[srgbGray];
}

// 打包回 ARGB_8888 (保留 Alpha 通道)
inline uint32_t pack8888(uint32_t originalColor, uint8_t srgbGray) {
    uint32_t alpha = originalColor & 0xFF000000;
    return alpha | (srgbGray << 16) | (srgbGray << 8) | srgbGray;
}

// 打包回 RGB_565
inline uint16_t pack565(uint8_t srgbGray) {
    int r5 = srgbGray >> 3;
    int g6 = srgbGray >> 2;
    int b5 = srgbGray >> 3;
    return static_cast<uint16_t>((r5 << 11) | (g6 << 5) | b5);
}

// --- 高精度线性空间 Floyd-Steinberg 误差扩散 ---

template <typename T>
void applyLinearFloydSteinberg(void* pixelsRaw, int width, int height, bool is8888) {
    T* pixels = static_cast<T*>(pixelsRaw);

    // 256 阶灰度量化步长 (E-ink 常用 16 阶灰度输出: STEP = 255 / 15 = 17)
    // 如果你需要纯黑白二值抖动，可以将 LEVEL 改为 2，STEP 改为 255
    const int LEVELS = 16;
    const int STEP = 255 / (LEVELS - 1);

    // 使用双行滑动误差缓冲区（以像素为单位），带左右边界保护
    std::vector<int> currRowErr(width + 2, 0);
    std::vector<int> nextRowErr(width + 2, 0);

    for (int y = 0; y < height; ++y) {
        std::fill(nextRowErr.begin(), nextRowErr.end(), 0);

        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            T originalColor = pixels[index];

            // 1. 提取线性空间灰度 (Linear Lum)
            int linearLum = is8888 ? getLinearLuminance8888(originalColor)
                                   : getLinearLuminance565(originalColor);

            // 2. 加上上一级传过来的扩散误差
            int adjustedLinear = linearLum + currRowErr[x + 1];
            adjustedLinear = CLAMP_255(adjustedLinear);

            // 3. 在线性空间下进行多阶量化
            int quantizedLinear = static_cast<int>(std::round(static_cast<float>(adjustedLinear) / STEP)) * STEP;
            quantizedLinear = CLAMP_255(quantizedLinear);

            // 4. 将量化后的线性灰度编码回 sRGB 空间写回内存
            uint8_t finalSrgb = Linear_to_sRGB[quantizedLinear];
            if (is8888) {
                pixels[index] = pack8888(originalColor, finalSrgb);
            } else {
                pixels[index] = pack565(finalSrgb);
            }

            // 5. 计算线性空间下的精确量化残差 (Error)
            int quantError = adjustedLinear - quantizedLinear;

            // 6. 无损 Floyd-Steinberg 误差扩散 (7/16, 3/16, 5/16, 1/16)
            // 右
            currRowErr[x + 2] += (quantError * 7) / 16;
            // 左下
            nextRowErr[x]     += (quantError * 3) / 16;
            // 下
            nextRowErr[x + 1] += (quantError * 5) / 16;
            // 右下
            nextRowErr[x + 2] += (quantError * 1) / 16;
        }

        // 交换滑动缓冲区
        std::swap(currRowErr, nextRowErr);
    }
}

// --- JNI 入口 ---

extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    // 确保 Gamma 2.2 LUT 表初始化完成
    initGammaTables();

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        LOGE("AndroidBitmap_getInfo failed");
        return;
    }

    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        LOGE("AndroidBitmap_lockPixels failed");
        return;
    }

    LOGI("Processing Dither (Linear Gamma 2.2): W=%d, H=%d, Format=%d", info.width, info.height, info.format);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        // 最佳 256 阶全彩输入
        applyLinearFloydSteinberg<uint32_t>(pixels, info.width, info.height, true);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        // 兼容 16 位输入（做无损 256 阶位深拉伸后处理）
        applyLinearFloydSteinberg<uint16_t>(pixels, info.width, info.height, false);
    } else {
        LOGE("Unsupported bitmap format: %d", info.format);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
