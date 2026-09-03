#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <vector>
#include <cstring>
#include <algorithm>

#define TAG "zyymeEink256"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

#define ERR_OFFSET 2048
#define ERR_RANGE (ERR_OFFSET * 2)

// --- 全局预计算查表 (LUT) ---
static int16_t LEVEL_TBL[ERR_RANGE];
static int16_t E7_TBL[ERR_RANGE];
static int16_t E5_TBL[ERR_RANGE];
static int16_t E3_TBL[ERR_RANGE];
static int16_t E1_TBL[ERR_RANGE];

static bool g_tables_initialized = false;

// 动态初始化查表（仅在第一次调用时执行一次）
static void initDitherTables() {
    if (g_tables_initialized) return;

    for (int i = 0; i < ERR_RANGE; i++) {
        int e = i - ERR_OFFSET;
        
        // 16阶灰度量化查表 (步长 17)
        int lvl = (e + 8) / 17;
        if (lvl < 0) lvl = 0;
        if (lvl > 15) lvl = 15;
        LEVEL_TBL[i] = lvl * 17;

        // 误差扩散系数查表 (避免运行时除法)
        E7_TBL[i] = (e * 7) / 16;
        E5_TBL[i] = (e * 5) / 16;
        E3_TBL[i] = (e * 3) / 16;
        E1_TBL[i] = (e) / 16;
    }
    g_tables_initialized = true;
}

// --- 像素访问辅助类 ---

// ARGB_8888 (32位)
struct Pixel8888 {
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        // 整数移位计算灰度: Y ≈ (77*R + 150*G + 29*B) >> 8
        return (77 * r + 150 * g + 29 * b) >> 8;
    }

    static inline uint32_t pack(uint32_t original, int grayVal) {
        uint32_t alpha = original & 0xFF000000;
        return alpha | (grayVal << 16) | (grayVal << 8) | grayVal;
    }
};

// RGB_565 (16位)
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

    static inline uint16_t pack(uint16_t original, int grayVal) {
        int r5 = grayVal >> 3;
        int g6 = grayVal >> 2;
        int b5 = grayVal >> 3;
        return (r5 << 11) | (g6 << 5) | b5;
    }
};

// --- 优化后的模板化抖动算法 ---
template <typename T, typename Handler>
void applyDitherTemplate(void* pixelsRaw, int width, int height, int stridePixels) {
    T* pixels = (T*)pixelsRaw;

    std::vector<int> currRowErr(width, 0);
    std::vector<int> nextRowErr(width, 0);

    for (int y = 0; y < height; ++y) {
        // 使用 stride 寻址，防止 Bitmap 内部行对齐填充导致的图像倾斜问题
        T* linePtr = pixels + y * stridePixels;

        for (int x = 0; x < width; ++x) {
            T originalColor = linePtr[x];

            // 1. 获取基础灰度
            int gray = Handler::getGray(originalColor);

            // 2. 加上上一像素扩散过来的误差
            int ge = gray + currRowErr[x];

            // 3. 查表获取量化灰度（替代原来的 std::round 浮点运算）
            int idx = ge + ERR_OFFSET;
            if (idx < 0) idx = 0;
            else if (idx >= ERR_RANGE) idx = ERR_RANGE - 1;

            int newGray = LEVEL_TBL[idx];

            // 4. 计算当前残余误差
            int err = ge - newGray;
            int eidx = err + ERR_OFFSET;
            if (eidx < 0) eidx = 0;
            else if (eidx >= ERR_RANGE) eidx = ERR_RANGE - 1;

            // 5. 查表扩散误差
            if (x + 1 < width) {
                currRowErr[x + 1] += E7_TBL[eidx];
            }
            if (y + 1 < height) {
                if (x - 1 >= 0) nextRowErr[x - 1] += E3_TBL[eidx];
                nextRowErr[x] += E5_TBL[eidx];
                if (x + 1 < width) nextRowErr[x + 1] += E1_TBL[eidx];
            }

            // 6. 写回像素
            linePtr[x] = Handler::pack(originalColor, newGray);
        }

        // 行误差轮换与清零
        currRowErr = nextRowErr;
        std::fill(nextRowErr.begin(), nextRowErr.end(), 0);
    }
}

// --- JNI 导出入口 ---
extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    initDitherTables();

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    LOGI("Dithering Bitmap: W=%d H=%d Stride=%d Format=%d", info.width, info.height, info.stride, info.format);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        // stride 是字节跨度，转换为 uint32_t 像素跨度
        int stridePixels = info.stride / sizeof(uint32_t);
        applyDitherTemplate<uint32_t, Pixel8888>(pixels, info.width, info.height, stridePixels);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        // stride 是字节跨度，转换为 uint16_t 像素跨度
        int stridePixels = info.stride / sizeof(uint16_t);
        applyDitherTemplate<uint16_t, Pixel565>(pixels, info.width, info.height, stridePixels);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
