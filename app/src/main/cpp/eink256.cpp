#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <android/log.h>

#define LOG_TAG "zyymeEink256"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define MIDTONE_SAMPLE_STEP 8
#define MIDTONE_THRESHOLD_PERCENT 0.04 

// --- 全局查找表 (LUT) ---
static bool g_tables_initialized = false;
static uint8_t LEVEL_TBL[256];
static uint8_t CONTRAST_LUT[256];
static uint8_t R5_TO_8[32];
static uint8_t G6_TO_8[64];

static void init_tables() {
    if (g_tables_initialized) return;

    for (int i = 0; i < 256; ++i) {
        LEVEL_TBL[i] = static_cast<uint8_t>(((i + 8) / 17) * 17);
    }

    // 温和 S 型对比度增强 LUT
    for (int i = 0; i < 256; ++i) {
        if (i <= 10) {
            CONTRAST_LUT[i] = 0;
        } else if (i >= 245) {
            CONTRAST_LUT[i] = 255;
        } else {
            double norm = (i - 10.0) / (245.0 - 10.0);
            double sigmoid = 1.0 / (1.0 + exp(-3.5 * (norm - 0.42)));
            double val = sigmoid * 255.0;
            if (val < 0.0) val = 0.0;
            if (val > 255.0) val = 255.0;
            CONTRAST_LUT[i] = static_cast<uint8_t>(val);
        }
    }

    for (int i = 0; i < 32; ++i) R5_TO_8[i] = (i << 3) | (i >> 2);
    for (int i = 0; i < 64; ++i) G6_TO_8[i] = (i << 2) | (i >> 4);

    g_tables_initialized = true;
}

// --- 高效灰度解包结构 ---
struct Pixel8888 {
    static inline uint8_t getGray(uint32_t color) {
        uint32_t r = (color >> 16) & 0xFF;
        uint32_t g = (color >> 8) & 0xFF;
        uint32_t b = color & 0xFF;
        return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
    }
    static inline uint32_t pack(uint32_t original, uint8_t grayVal) {
        return (original & 0xFF000000) | (grayVal << 16) | (grayVal << 8) | grayVal;
    }
};

struct Pixel565 {
    static inline uint8_t getGray(uint16_t color) {
        uint32_t r8 = R5_TO_8[(color >> 11) & 0x1F];
        uint32_t g8 = G6_TO_8[(color >> 5) & 0x3F];
        uint32_t b8 = R5_TO_8[color & 0x1F];
        return static_cast<uint8_t>((77 * r8 + 150 * g8 + 29 * b8) >> 8);
    }
    static inline uint16_t pack(uint16_t original, uint8_t grayVal) {
        uint16_t r5 = grayVal >> 3;
        uint16_t g6 = grayVal >> 2;
        uint16_t b5 = grayVal >> 3;
        return (r5 << 11) | (g6 << 5) | b5;
    }
};

// 文本识别快速采样
template <typename T, typename Handler>
bool shouldDitherImage(const void* pixelsRaw, int width, int height) {
    const T* pixels = static_cast<const T*>(pixelsRaw);
    int midtoneCount = 0;
    int sampledCount = 0;

    for (int y = 0; y < height; y += MIDTONE_SAMPLE_STEP) {
        int rowOffset = y * width;
        for (int x = 0; x < width; x += MIDTONE_SAMPLE_STEP) {
            uint8_t gray = Handler::getGray(pixels[rowOffset + x]);
            sampledCount++;
            if (gray > 24 && gray < 231) midtoneCount++;
        }
    }

    return (static_cast<float>(midtoneCount) / sampledCount) >= MIDTONE_THRESHOLD_PERCENT;
}

// 高性能抖动主逻辑
template <typename T, typename Handler>
void applyHighQualityDither(void* pixelsRaw, int width, int height) {
    if (!shouldDitherImage<T, Handler>(pixelsRaw, width, height)) {
        LOGI("Detected text-only layout, skipped dithering.");
        return;
    }

    T* pixels = static_cast<T*>(pixelsRaw);

    // 预分配整屏灰度缓冲区（连续内存，极大提升 Cache 命中率）
    std::vector<uint8_t> grayBuffer(width * height);
    for (int i = 0; i < width * height; ++i) {
        grayBuffer[i] = Handler::getGray(pixels[i]);
    }

    // 2行残差缓冲区
    std::vector<int16_t> err_curr(width + 2, 0);
    std::vector<int16_t> err_next(width + 2, 0);

    for (int y = 0; y < height; ++y) {
        int row_offset = y * width;
        std::fill(err_next.begin(), err_next.end(), 0);

        if ((y & 1) == 0) {
            // 正向：完全展开的无分支循环
            for (int x = 0; x < width; ++x) {
                int idx = row_offset + x;
                uint8_t src_gray = grayBuffer[idx];
                uint8_t contrast_gray = CONTRAST_LUT[src_gray];

                int raw_gray = contrast_gray + err_curr[x + 1];
                int clamped = (raw_gray < 0) ? 0 : ((raw_gray > 255) ? 255 : raw_gray);

                uint8_t qval = LEVEL_TBL[clamped];
                pixels[idx] = Handler::pack(pixels[idx], qval);

                // 基于灰度缓冲区的快速边缘检测
                bool is_edge = false;
                if (x > 0 && x < width - 1) {
                    int diff = std::abs(static_cast<int>(src_gray) - grayBuffer[idx - 1]);
                    if (diff > 50) is_edge = true;
                }

                if (!is_edge) {
                    int err = raw_gray - qval;
                    // 残影控制：微小误差门限过滤 (Noise Thresholding)
                    if (std::abs(err) > 2) {
                        err_curr[x + 2] += (err * 7) >> 4;
                        err_next[x + 0] += (err * 3) >> 4;
                        err_next[x + 1] += (err * 5) >> 4;
                        err_next[x + 2] += (err * 1) >> 4;
                    }
                }
            }
        } else {
            // 反向：无分支循环
            for (int x = width - 1; x >= 0; --x) {
                int idx = row_offset + x;
                uint8_t src_gray = grayBuffer[idx];
                uint8_t contrast_gray = CONTRAST_LUT[src_gray];

                int raw_gray = contrast_gray + err_curr[x + 1];
                int clamped = (raw_gray < 0) ? 0 : ((raw_gray > 255) ? 255 : raw_gray);

                uint8_t qval = LEVEL_TBL[clamped];
                pixels[idx] = Handler::pack(pixels[idx], qval);

                bool is_edge = false;
                if (x > 0 && x < width - 1) {
                    int diff = std::abs(static_cast<int>(src_gray) - grayBuffer[idx - 1]);
                    if (diff > 50) is_edge = true;
                }

                if (!is_edge) {
                    int err = raw_gray - qval;
                    if (std::abs(err) > 2) {
                        err_curr[x + 0] += (err * 7) >> 4;
                        err_next[x + 2] += (err * 3) >> 4;
                        err_next[x + 1] += (err * 5) >> 4;
                        err_next[x + 0] += (err * 1) >> 4;
                    }
                }
            }
        }

        std::swap(err_curr, err_next);
    }
}

// JNI 导出接口
extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    init_tables();

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyHighQualityDither<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyHighQualityDither<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
