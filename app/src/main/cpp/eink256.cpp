#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <cstdint>
#include <algorithm>

// 4x4 Bayer 抖动矩阵 (范围 0~15)
static const uint8_t BAYER_PATTERN_4X4[4][4] = {
    {  0,  8,  2, 10 },
    { 12,  4, 14,  6 },
    {  3, 11,  1,  9 },
    { 15,  7, 13,  5 }
};

// 预计算的灰度量化查找表 (LUT)：输入 0~255，输出 16 阶灰度 (0, 17, 34, ..., 255)
static uint8_t GRAY_QUANT_LUT[256];
static bool g_lut_inited = false;

static void init_lut() {
    if (g_lut_inited) return;
    for (int i = 0; i < 256; ++i) {
        // 量化到 16 阶 (STEP = 17)
        GRAY_QUANT_LUT[i] = ((i + 8) / 17) * 17;
    }
    g_lut_inited = true;
}

// 快速 ARGB8888 灰度 + Bayer 抖动
static inline void process_rgba8888_bayer(uint8_t* linePtr, int width, int y) {
    uint32_t* pixels = (uint32_t*)linePtr;
    const uint8_t* bayerRow = BAYER_PATTERN_4X4[y & 3];

    for (int x = 0; x < width; ++x) {
        uint32_t color = pixels[x];
        
        // 快速整数灰度: (77R + 150G + 29B) >> 8
        uint32_t r = (color >> 16) & 0xFF;
        uint32_t g = (color >> 8) & 0xFF;
        uint32_t b = color & 0xFF;
        int gray = (77 * r + 150 * g + 29 * b) >> 8;

        // 加上 Bayer 矩阵偏移 (值域 -8 ~ +7)
        int threshold = bayerRow[x & 3] - 8;
        int ditheredGray = gray + threshold;
        
        // Clamp 0-255
        ditheredGray = ditheredGray < 0 ? 0 : (ditheredGray > 255 ? 255 : ditheredGray);

        // 查表获取 16 阶灰度
        uint8_t finalGray = GRAY_QUANT_LUT[ditheredGray];

        // 写回 (保留 Alpha)
        pixels[x] = (color & 0xFF000000) | (finalGray << 16) | (finalGray << 8) | finalGray;
    }
}

// 快速 RGB565 灰度 + Bayer 抖动
static inline void process_rgb565_bayer(uint8_t* linePtr, int width, int y) {
    uint16_t* pixels = (uint16_t*)linePtr;
    const uint8_t* bayerRow = BAYER_PATTERN_4X4[y & 3];

    for (int x = 0; x < width; ++x) {
        uint16_t color = pixels[x];

        // 解包 RGB565 并快速扩展到 8bit
        int r5 = (color >> 11) & 0x1F;
        int g6 = (color >> 5) & 0x3F;
        int b5 = color & 0x1F;

        int r8 = (r5 << 3) | (r5 >> 2);
        int g8 = (g6 << 2) | (g6 >> 4);
        int b8 = (b5 << 3) | (b5 >> 2);

        int gray = (77 * r8 + 150 * g8 + 29 * b8) >> 8;

        int threshold = bayerRow[x & 3] - 8;
        int ditheredGray = gray + threshold;
        ditheredGray = ditheredGray < 0 ? 0 : (ditheredGray > 255 ? 255 : ditheredGray);

        uint8_t finalGray = GRAY_QUANT_LUT[ditheredGray];

        // 打包回 RGB565
        int r5_out = finalGray >> 3;
        int g6_out = finalGray >> 2;
        int b5_out = finalGray >> 3;

        pixels[x] = (r5_out << 11) | (g6_out << 5) | b5_out;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    init_lut();

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    uint8_t* linePtr = (uint8_t*)pixels;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        for (int y = 0; y < info.height; ++y) {
            process_rgba8888_bayer(linePtr, info.width, y);
            linePtr += info.stride; // 安全处理内存对齐
        }
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        for (int y = 0; y < info.height; ++y) {
            process_rgb565_bayer(linePtr, info.width, y);
            linePtr += info.stride;
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
