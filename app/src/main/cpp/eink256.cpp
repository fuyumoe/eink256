#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

// --- 像素访问辅助类 ---
struct Pixel8888 {
    static inline int getGray(uint32_t color) {
        int r = (color >> 16) & 0xFF;
        int g = (color >> 8) & 0xFF;
        int b = color & 0xFF;
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

// 预计算的 16阶 灰度量化查找表 (LUT)：替换掉 std::round 浮点运算
static uint8_t QUANT_LUT[512]; // 支持 [-128, 383] 的溢出灰度值
static bool g_lut_inited = false;

static void init_quant_lut() {
    if (g_lut_inited) return;
    // STEP = 17 (255 / 15)
    for (int i = -128; i < 384; ++i) {
        int clamped = CLAMP(i);
        // 原版的 std::round((float)gray / 17) * 17 逻辑
        int quantized = ((clamped + 8) / 17) * 17;
        QUANT_LUT[i + 128] = CLAMP(quantized);
    }
    g_lut_inited = true;
}

// 快速 16 阶量化查找
static inline int fastQuantize16(int gray) {
    // 防止极其罕见的极大/极小误差越界
    if (gray < -128) return 0;
    if (gray > 383) return 255;
    return QUANT_LUT[gray + 128];
}

// --- 修正后的 Floyd-Steinberg 抖动模板 ---
template <typename T, typename Handler>
void applyDitherTemplateFast(uint8_t* pixelsRaw, int width, int height, uint32_t stride) {
    // 使用指针动态申请内存，避免 std::vector 深拷贝
    int* currRowErr = (int*)calloc(width, sizeof(int));
    int* nextRowErr = (int*)calloc(width, sizeof(int));

    for (int y = 0; y < height; ++y) {
        // 使用 stride 正确处理 Android Bitmap 内存对齐
        T* line = (T*)(pixelsRaw + y * stride);

        for (int x = 0; x < width; ++x) {
            T originalColor = line[x];

            // 1. 获取灰度并叠加上一行/上一像素传过来的误差
            int gray = Handler::getGray(originalColor) + currRowErr[x];

            // 2. 整数 LUT 快速量化 (100% 等价于原来的 std::round((float)gray / STEP) * STEP)
            int newGray = fastQuantize16(gray);

            // 3. 计算误差
            int quantError = gray - newGray;

            // 4. Floyd-Steinberg 误差扩散 (用位移代替除法，速度大幅提升)
            // 右: 7/16
            if (x + 1 < width) {
                currRowErr[x + 1] += (quantError * 7) / 16;
            }

            // 下一行扩散
            if (y + 1 < height) {
                // 左下: 3/16
                if (x > 0) nextRowErr[x - 1] += (quantError * 3) / 16;
                // 下: 5/16
                nextRowErr[x] += (quantError * 5) / 16;
                // 右下: 1/16
                if (x + 1 < width) nextRowErr[x + 1] += quantError / 16;
            }

            // 5. 写回内存
            line[x] = Handler::pack(originalColor, newGray);
        }

        // 行结束：完全交换指针，并将【下一行】清零准备下一次循环（关键修复！）
        int* temp = currRowErr;
        currRowErr = nextRowErr; // 现在 currRowErr 持有了刚才累积的 nextRowErr 数据
        nextRowErr = temp;       // nextRowErr 指向了废弃的旧内存
        memset(nextRowErr, 0, width * sizeof(int)); // 正确：清空废弃的内存，作为下下行的累加器
    }

    free(currRowErr);
    free(nextRowErr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_zyyme_eink256_Eink256Native_ditherBitmap(JNIEnv* env, jclass clazz, jobject bitmap) {
    init_quant_lut();

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    __android_log_print(ANDROID_LOG_INFO, "zyymeEink256", "Dithering Fast FS: W=%d H=%d Format=%d Stride=%d", 
                        info.width, info.height, info.format, info.stride);

    uint8_t* pixelsRaw = (uint8_t*)pixels;

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        applyDitherTemplateFast<uint32_t, Pixel8888>(pixelsRaw, info.width, info.height, info.stride);
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        applyDitherTemplateFast<uint16_t, Pixel565>(pixelsRaw, info.width, info.height, info.stride);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
