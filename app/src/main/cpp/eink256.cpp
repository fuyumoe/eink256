#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <android/log.h>

#define LOG_TAG "zyymeEink256"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// 采样检测配置：智能识别纯文本页，防止文字笔画产生“麻点”
#define MIDTONE_SAMPLE_STEP 8
#define MIDTONE_THRESHOLD_PERCENT 0.04 // 中间调像素占比低于 4% 则判定为纯文本页

// --- 查找表配置 ---
static bool g_tables_initialized = false;
static uint8_t LEVEL_TBL[256];

static void init_tables() {
    if (g_tables_initialized) return;
    // 16阶灰度精准映射（0, 17, 34, ..., 255）
    for (int i = 0; i < 256; ++i) {
        LEVEL_TBL[i] = static_cast<uint8_t>(((i + 8) / 17) * 17);
    }
    g_tables_initialized = true;
}

// --- 像素处理 Handler 类 ---

struct Pixel8888 {
    // 采用更贴近人眼视觉感知的 Luma 灰度权重 (0.299R + 0.587G + 0.114B)
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
        int r5 = (color >> 11) & 0x1F;
        int g6 = (color >> 5) & 0x3F;
        int b5 = color & 0x1F;
        
        int r8 = (r5 << 3) | (r5 >> 2);
        int g8 = (g6 << 2) | (g6 >> 4);
        int b8 = (b5 << 3) | (b5 >> 2);
        
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
            // 灰度在 24~231 之间判定为包含图像细节的中间调
            if (gray > 24 && gray < 231) {
                midtoneCount++;
            }
        }
    }

    float ratio = static_cast<float>(midtoneCount) / static_cast<float>(sampledCount);
    return ratio >= MIDTONE_THRESHOLD_PERCENT;
}

// --- 蛇形遍历（Serpentine）Floyd-Steinberg 高细腻度抖动 ---
template <typename T, typename Handler>
void applyHighQualityDither(void* pixelsRaw, int width, int height) {
    // 纯文本页跳过抖动，维持文字绝对干脆锐利
    if (!shouldDitherImage<T, Handler>(pixelsRaw, width, height)) {
        LOGI("Detected text-only layout, preserving crisp vector font rendering.");
        return;
    }

    T* pixels = static_cast<T*>(pixelsRaw);

    // 误差缓冲区，使用 int 类型防止累加溢出
    std::vector<int> err_curr(width + 2, 0);
    std::vector<int> err_next(width + 2, 0);

    for (int y = 0; y < height; ++y) {
        int row_offset = y * width;
        std::fill(err_next.begin(), err_next.end(), 0);

        bool left_to_right = (y % 2 == 0); // 偶数行从左到右，奇数行从右到左

        if (left_to_right) {
            // 从左向右扫描
            for (int x = 0; x < width; ++x) {
                int idx = row_offset + x;
                T originalColor = pixels[idx];

                int raw_gray = Handler::getGray(originalColor) + err_curr[x + 1];
                int clamped = (raw_gray < 0) ? 0 : ((raw_gray > 255) ? 255 : raw_gray);

                uint8_t qval = LEVEL_TBL[clamped];
                pixels[idx] = Handler::pack(originalColor, qval);

                int err = raw_gray - qval; // 使用 raw_gray 计算真实残差，极大丰富暗部与高光细节

                // 扩散误差 (向右, 左下, 下, 右下)
                err_curr[x + 2] += err * 7 / 16;
                err_next[x + 0] += err * 3 / 16;
                err_next[x + 1] += err * 5 / 16;
                err_next[x + 2] += err * 1 / 16;
            }
        } else {
            // 从右向左扫描（消除方向性网纹）
            for (int x = width - 1; x >= 0; --x) {
                int idx = row_offset + x;
                T originalColor = pixels[idx];

                int raw_gray = Handler::getGray(originalColor) + err_curr[x + 1];
                int clamped = (raw_gray < 0) ? 0 : ((raw_gray > 255) ? 255 : raw_gray);

                uint8_t qval = LEVEL_TBL[clamped];
                pixels[idx] = Handler::pack(originalColor, qval);

                int err = raw_gray - qval;

                // 反向扩散误差 (向左, 右下, 下, 左下)
                err_curr[x + 0] += err * 7 / 16;
                err_next[x + 2] += err * 3 / 16;
                err_next[x + 1] += err * 5 / 16;
                err_next[x + 0] += err * 1 / 16;
            }
        }

        std::swap(err_curr, err_next);
    }
}

// --- JNI 接口入口 ---
extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    init_tables();

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    LOGI("Dithering Bitmap (High Quality Mode): W=%d H=%d Format=%d", info.width, info.height, info.format);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyHighQualityDither<uint32_t, Pixel8888>(pixels, info.width, info.height);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyHighQualityDither<uint16_t, Pixel565>(pixels, info.width, info.height);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
