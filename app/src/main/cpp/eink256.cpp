#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <android/log.h>

#define LOG_TAG "zyymeEink256"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// 智能采样配置 (对应 Lua 代码中的 midtone 判断逻辑)
#define MIDTONE_SAMPLE_STEP 8
#define MIDTONE_THRESHOLD_PERCENT 0.05 // 中间调占比低于 5% 判定为纯文本页，跳过抖动

// --- 预计算查找表 (LUT) ---
static bool g_tables_initialized = false;

// 1. 16阶量化表 (0-255 -> 0, 17, 34, ..., 255)
static uint8_t LEVEL_TBL[256];

// 2. Floyd-Steinberg 误差扩散查找表 (误差范围 -255 ~ +255，偏移 256)
static int16_t E7_TBL[512]; // * 7 / 16
static int16_t E3_TBL[512]; // * 3 / 16
static int16_t E5_TBL[512]; // * 5 / 16
static int16_t E1_TBL[512]; // * 1 / 16

// 3. RGB565 分量扩展与打包查找表 (大幅提升 RGB565 处理速度)
static uint8_t R5_TO_8[32];
static uint8_t G6_TO_8[64];

static void init_tables() {
    if (g_tables_initialized) return;

    // 初始化 16 阶量化表 (步长 17)
    for (int i = 0; i < 256; ++i) {
        LEVEL_TBL[i] = static_cast<uint8_t>(((i + 8) / 17) * 17);
    }

    // 初始化误差扩散表 (消灭乘除法运算)
    for (int e = -255; e <= 255; ++e) {
        int idx = e + 256;
        E7_TBL[idx] = static_cast<int16_t>(e * 7 / 16);
        E3_TBL[idx] = static_cast<int16_t>(e * 3 / 16);
        E5_TBL[idx] = static_cast<int16_t>(e * 5 / 16);
        E1_TBL[idx] = static_cast<int16_t>(e * 1 / 16);
    }

    // 初始化 RGB565 颜色查找表
    for (int i = 0; i < 32; ++i) R5_TO_8[i] = (i << 3) | (i >> 2);
    for (int i = 0; i < 64; ++i) G6_TO_8[i] = (i << 2) | (i >> 4);

    g_tables_initialized = true;
}

// --- 像素处理 Handler 类 ---

struct Pixel8888 {
    // 心理学灰度公式: 0.299R + 0.587G + 0.114B (整数乘法替代浮点)
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

// --- 智能文本/中间调过滤检测 (对应 Lua 的 hasImageContent 逻辑) ---
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
            // 灰度在 24~231 之间判定为中间调（图像/照片区域）
            if (gray > 24 && gray < 231) {
                midtoneCount++;
            }
        }
    }

    float ratio = (float)midtoneCount / (float)sampledCount;
    // 如果中间调占比低于阈值，说明是纯黑白字文档，直接跳过抖动
    return ratio >= MIDTONE_THRESHOLD_PERCENT;
}

// --- 核心优化抖动算法 ---
template <typename T, typename Handler>
void applyDitherTemplate(void* pixelsRaw, int width, int height) {
    // 1. 采样检测：如果是纯文本页，跳过抖动，保持原生矢量文字锐利度
    if (!shouldDitherImage<T, Handler>(pixelsRaw, width, height)) {
        LOGI("Detected text-only page, skipping dithering for sharp font edges.");
        return;
    }

    T* pixels = static_cast<T*>(pixelsRaw);

    // 2. 带有 1 像素边界 padding 的双行滑动误差缓冲区
    std::vector<int16_t> err_curr(width + 2, 0);
    std::vector<int16_t> err_next(width + 2, 0);

    for (int y = 0; y < height; ++y) {
        int row_offset = y * width;
        std::fill(err_next.begin(), err_next.end(), 0);

        for (int x = 0; x < width; ++x) {
            int idx = row_offset + x;
            T originalColor = pixels[idx];

            // 获取灰度并叠加传输过来的误差
            int gray = Handler::getGray(originalColor) + err_curr[x + 1];

            // 安全 Clamp 范围限定到 0-255
            uint8_t clamped = (gray < 0) ? 0 : ((gray > 255) ? 255 : gray);

            // 16 阶 LUT 直写量化
            uint8_t qval = LEVEL_TBL[clamped];

            // 写回内存
            pixels[idx] = Handler::pack(originalColor, qval);

            // 计算误差并映射到查找表偏移索引 (加 256)
            int err_idx = (clamped - qval) + 256;

            // 高性能 LUT 扩散误差 (避免乘除法)
            err_curr[x + 2] += E7_TBL[err_idx];
            err_next[x + 0] += E3_TBL[err_idx];
            err_next[x + 1] += E5_TBL[err_idx];
            err_next[x + 2] += E1_TBL[err_idx];
        }

        // 交换当前行与下一行误差
        std::swap(err_curr, err_next);
    }
}

// --- JNI 导出入口 ---
extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    init_tables(); // 初始化 LUT

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    LOGI("Dithering Bitmap: W=%d H=%d Format=%d", info.width, info.height, info.format);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyDitherTemplate<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyDitherTemplate<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
