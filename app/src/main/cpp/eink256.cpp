#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <android/log.h>

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

// --- 预计算查找表定义 (消除除法，对齐 Lua 查找表逻辑) ---
static const int ERR_OFFSET = 2048;
static const int ERR_RANGE = ERR_OFFSET * 2;

static int16_t LEVEL_TBL[ERR_RANGE];
static int16_t E7_TBL[ERR_RANGE];
static int16_t E5_TBL[ERR_RANGE];
static int16_t E3_TBL[ERR_RANGE];
static int16_t E1_TBL[ERR_RANGE];

static bool g_lutInitialized = false;

// 初始化抖动与误差查找表
static void initDitherTables() {
    if (g_lutInitialized) return;
    for (int i = 0; i < ERR_RANGE; ++i) {
        int e = i - ERR_OFFSET;
        // 16阶灰度，步长17: (e + 8) / 17
        int lvl = (e + 8) / 17;
        if (lvl < 0) lvl = 0;
        else if (lvl > 15) lvl = 15;
        
        LEVEL_TBL[i] = lvl * 17;
        E7_TBL[i] = (e * 7) / 16;
        E5_TBL[i] = (e * 5) / 16;
        E3_TBL[i] = (e * 3) / 16;
        E1_TBL[i] = e / 16;
    }
    g_lutInitialized = true;
}

// --- 像素访问辅助类 ---

struct Pixel8888 {
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
        // BT.601 精确感知灰度: (77R + 150G + 29B) >> 8
        return (77 * r + 150 * g + 29 * b) >> 8;
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

        // 位扩展 565 到 8位
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

// --- 模板化抖动算法 ---
template <typename T, typename Handler>
void applyDitherTemplate(void* pixelsRaw, int width, int height, int stridePixels) {
    T* pixels = (T*)pixelsRaw;

    // 采用固定 C 数组 / std::vector 避免反复分配，提升性能
    std::vector<int> currRowErr(width, 0);
    std::vector<int> nextRowErr(width, 0);

    const int wm1 = width - 1;
    const int hm1 = height - 1;

    for (int y = 0; y < height; ++y) {
        // 【关键点 1】使用真实的 stridePixels 计算行起始指针，解决对齐引起的断层
        T* rowPixels = pixels + y * stridePixels;

        for (int x = 0; x < width; ++x) {
            T originalColor = rowPixels[x];
            
            // 1. 获取基础灰度
            int gray = Handler::getGray(originalColor);

            // 2. 加上扩散误差（不提前 CLAMP）
            int ge = gray + currRowErr[x];

            // 3. 查表获取新灰度（与 Lua 算法完全对齐，消除浮点与除法）
            int idx = ge + ERR_OFFSET;
            if (idx < 0) idx = 0;
            else if (idx > ERR_RANGE - 1) idx = ERR_RANGE - 1;

            int newGray = LEVEL_TBL[idx];

            // 【关键点 2】用包含误差的 ge 减去 newGray 计算未损失的残差
            int err = ge - newGray;

            // 4. 写入内存
            rowPixels[x] = Handler::pack(originalColor, newGray);

            // 5. 扩散残差 (查表加速，无溢出强砍)
            int eidx = err + ERR_OFFSET;
            if (eidx < 0) eidx = 0;
            else if (eidx > ERR_RANGE - 1) eidx = ERR_RANGE - 1;

            if (x < wm1) {
                currRowErr[x + 1] += E7_TBL[eidx];
            }
            if (y < hm1) {
                if (x > 0) nextRowErr[x - 1] += E3_TBL[eidx];
                nextRowErr[x] += E5_TBL[eidx];
                if (x < wm1) nextRowErr[x + 1] += E1_TBL[eidx];
            }
        }

        // 行交换与清零
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

    initDitherTables();

    // 【关键点 3】计算以像素为单位的 Stride，避免硬件内存对其导致的图片错位/渐变断层
    int stridePixels = 0;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        stridePixels = info.stride / 4;
        applyDitherTemplate<uint32_t, Pixel8888>(pixels, info.width, info.height, stridePixels);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        stridePixels = info.stride / 2;
        applyDitherTemplate<uint16_t, Pixel565>(pixels, info.width, info.height, stridePixels);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
