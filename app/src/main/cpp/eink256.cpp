#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <android/log.h>

#define LOG_TAG "zyymeEink256"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// 采样检测配置：智能识别纯文本页，防止纯文本产生麻点
#define MIDTONE_SAMPLE_STEP 8
#define MIDTONE_THRESHOLD_PERCENT 0.04 

// --- 全局查找表 (LUT) ---
static bool g_tables_initialized = false;

// 1. 16阶灰度精准映射表 (0, 17, 34, ..., 255)
static uint8_t LEVEL_TBL[256];

// 2. S型伽马/对比度拉伸表 (切断双线性插值带来的伪灰阶)
static uint8_t CONTRAST_LUT[256];

// 3. RGB565 转换查找表
static uint8_t R5_TO_8[32];
static uint8_t G6_TO_8[64];

static void init_tables() {
    if (g_tables_initialized) return;

    // 初始化 16 阶量化表
    for (int i = 0; i < 256; ++i) {
        LEVEL_TBL[i] = static_cast<uint8_t>(((i + 8) / 17) * 17);
    }

    // 初始化 S 型对比度增强 LUT
    // 逻辑：压暗 < 45 的深灰为 0；拉亮 > 215 的浅灰为 255；中间调做 S-Curve 提升对比度
    for (int i = 0; i < 256; ++i) {
        if (i < 45) {
            CONTRAST_LUT[i] = 0;
        } else if (i > 215) {
            CONTRAST_LUT[i] = 255;
        } else {
            double norm = (i - 45.0) / (215.0 - 45.0); // 归一化到 0~1
            double sigmoid = 1.0 / (1.0 + exp(-6.0 * (norm - 0.5))); // Sigmoid 曲线
            double val = sigmoid * 255.0;
            if (val < 0.0) val = 0.0;
            if (val > 255.0) val = 255.0;
            CONTRAST_LUT[i] = static_cast<uint8_t>(val);
        }
    }

    // 初始化 RGB565 解码查找表
    for (int i = 0; i < 32; ++i) R5_TO_8[i] = (i << 3) | (i >> 2);
    for (int i = 0; i < 64; ++i) G6_TO_8[i] = (i << 2) | (i >> 4);

    g_tables_initialized = true;
}

// --- 像素处理 Handler 类 ---

struct Pixel8888 {
    // Luma 灰度权重 (0.299R + 0.587G + 0.114B)
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        return (77 * r + 150 * g + 29 * b) >> 8;
    }

    static inline uint32_t pack(uint32_t original, uint8_t grayVal) {
        return (original & 0xFF000000) | (grayVal << 16) | (grayVal << 8) | grayVal;
    }
};

struct Pixel565 {
    static inline int getGray(uint16_t color) {
        int r8 = R5_TO_8[(color >> 11) & 0x1F];
        int g8 = G6_TO_8[(color >> 5) & 0x3F];
        int b8 = R5_TO_8[color & 0x1F];
        return (77 * r8 + 150 * g8 + 29 * b8) >> 8;
    }

    static inline uint16_t pack(uint16_t original, uint8_t grayVal) {
        int r5 = grayVal >> 3;
        int g6 = grayVal >> 2;
        int b5 = grayVal >> 3;
        return (r5 << 11) | (g6 << 5) | b5;
    }
};

// --- 智能文本识别：判断是否需要抖动 ---
template <typename T, typename Handler>
bool shouldDitherImage(void* pixelsRaw, int width, int height) {
    T* pixels = static_cast<T*>(pixelsRaw);
    int midtoneCount = 0;
    int sampledCount = 0;

    for (int y = 0; y < height; y += MIDTONE_SAMPLE_STEP) {
        int rowOffset = y * width;
        for (int x = 0; x < width; x += MIDTONE_SAMPLE_STEP) {
            int gray = Handler::getGray(pixels[rowOffset + x]);
            sampledCount++;
            if (gray > 24 && gray < 231) {
                midtoneCount++;
            }
        }
    }

    float ratio = static_cast<float>(midtoneCount) / static_cast<float>(sampledCount);
    return ratio >= MIDTONE_THRESHOLD_PERCENT;
}

// --- 蛇形遍历 + 边缘保护 + S-Curve 复合抖动算法 ---
template <typename T, typename Handler>
void applyHighQualityDither(void* pixelsRaw, int width, int height) {
    if (!shouldDitherImage<T, Handler>(pixelsRaw, width, height)) {
        LOGI("Detected text-only layout, preserving crisp vector font rendering.");
        return;
    }

    T* pixels = static_cast<T*>(pixelsRaw);

    std::vector<int> err_curr(width + 2, 0);
    std::vector<int> err_next(width + 2, 0);

    for (int y = 0; y < height; ++y) {
        int row_offset = y * width;
        std::fill(err_next.begin(), err_next.end(), 0);

        bool left_to_right = (y % 2 == 0); // 偶数行从左向右，奇数行从右向左（蛇形遍历）

        if (left_to_right) {
            // --- 正向扫描（从左向右） ---
            for (int x = 0; x < width; ++x) {
                int idx = row_offset + x;
                T originalColor = pixels[idx];

                // 1. 获取基础灰度并进行 S-Curve 预处理
                int src_gray = Handler::getGray(originalColor);
                int contrast_gray = CONTRAST_LUT[src_gray];

                // 2. 叠加抖动误差
                int raw_gray = contrast_gray + err_curr[x + 1];
                int clamped = (raw_gray < 0) ? 0 : ((raw_gray > 255) ? 255 : raw_gray);

                // 3. 量化并写回像素
                uint8_t qval = LEVEL_TBL[clamped];
                pixels[idx] = Handler::pack(originalColor, qval);

                // 4. 高频边缘检测（Edge Protection）：判定是否为文字/线条的剧烈边缘
                bool is_edge = false;
                if (x > 0 && x < width - 1) {
                    int left = Handler::getGray(pixels[idx - 1]);
                    int right = Handler::getGray(pixels[idx + 1]);
                    if (std::abs(src_gray - left) > 55 || std::abs(src_gray - right) > 55) {
                        is_edge = true;
                    }
                }

                // 5. 仅在非高频边缘区域扩散误差，文字边缘禁扩散以防生成杂点网纹
                if (!is_edge) {
                    int err = raw_gray - qval;
                    err_curr[x + 2] += err * 7 / 16;
                    err_next[x + 0] += err * 3 / 16;
                    err_next[x + 1] += err * 5 / 16;
                    err_next[x + 2] += err * 1 / 16;
                }
            }
        } else {
            // --- 反向扫描（从右向左，消除结构性斜线） ---
            for (int x = width - 1; x >= 0; --x) {
                int idx = row_offset + x;
                T originalColor = pixels[idx];

                int src_gray = Handler::getGray(originalColor);
                int contrast_gray = CONTRAST_LUT[src_gray];

                int raw_gray = contrast_gray + err_curr[x + 1];
                int clamped = (raw_gray < 0) ? 0 : ((raw_gray > 255) ? 255 : raw_gray);

                uint8_t qval = LEVEL_TBL[clamped];
                pixels[idx] = Handler::pack(originalColor, qval);

                bool is_edge = false;
                if (x > 0 && x < width - 1) {
                    int left = Handler::getGray(pixels[idx - 1]);
                    int right = Handler::getGray(pixels[idx + 1]);
                    if (std::abs(src_gray - left) > 55 || std::abs(src_gray - right) > 55) {
                        is_edge = true;
                    }
                }

                if (!is_edge) {
                    int err = raw_gray - qval;
                    err_curr[x + 0] += err * 7 / 16;
                    err_next[x + 2] += err * 3 / 16;
                    err_next[x + 1] += err * 5 / 16;
                    err_next[x + 0] += err * 1 / 16;
                }
            }
        }

        std::swap(err_curr, err_next);
    }
}

// --- JNI 导出接口 ---
extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    init_tables();

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    LOGI("Dithering Bitmap (Anti-blur Mode): W=%d H=%d Format=%d", info.width, info.height, info.format);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyHighQualityDither<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyHighQualityDither<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
