#pragma once

#include "stb_image_write.h"
#include "tinyexr.h"

#include "utils/logger.h"

#include <vector>
#include <stdio.h>

namespace ntwr {

enum OutputImageFormat {
    OUTPUT_IMAGE_FORMAT_PNG,
    OUTPUT_IMAGE_FORMAT_EXR,
};

enum ExrCompression {
    EXR_COMPRESSION_ZIP,  // Good general purpose.
    EXR_COMPRESSION_PIZ,  // Good with noisy images.
    EXR_COMPRESSION_RLE,  // Good with large areas of identical color.
    EXR_COMPRESSION_NONE
};

template <typename T>
struct Bitmap {
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_channels;
    std::vector<T> m_image_data;
};

inline Bitmap<float> bitmap_from_vector(const std::vector<float> &pixel_data,
                                        uint32_t width,
                                        uint32_t height,
                                        uint32_t channels,
                                        float rgb_normalization = 1.f)
{
    Bitmap<float> exr_bitmap;
    exr_bitmap.m_width      = width;
    exr_bitmap.m_height     = height;
    exr_bitmap.m_channels   = channels;
    exr_bitmap.m_image_data = pixel_data;
    for (size_t i = 0; i < width * height; ++i) {
        for (size_t c = 0; c < channels; ++c) {
            if (c != 3)
                exr_bitmap.m_image_data[i * channels + c] /= rgb_normalization;
        }
    }
    return exr_bitmap;
}

inline Bitmap<float> load_exr(const char *filename, int required_width = -1, int required_height = -1)
{
    Bitmap<float> exr_bitmap;
    float *data;  // width * height * RGBA
    int width, height;
    const char *err = nullptr;

    int ret = LoadEXR(&data, &width, &height, filename, &err);
    bool size_mismatch =
        (required_width != -1 && required_width != width) || (required_height != -1 && required_height != height);
    if (ret != TINYEXR_SUCCESS || size_mismatch) {
        if (size_mismatch) {
            logger(LogLevel::Warn,
                   "Required size not matched, expected %dx%d got %dx%d, returning black image",
                   required_width,
                   required_height,
                   width,
                   height);
        } else {
            logger(LogLevel::Error, "Failed loading EXR %s, returning black image", filename);
        }
        exr_bitmap.m_width      = required_width != -1 ? (uint32_t)required_width : 1;
        exr_bitmap.m_height     = required_height != -1 ? (uint32_t)required_height : 1;
        exr_bitmap.m_channels   = 4;
        exr_bitmap.m_image_data = std::vector<float>(exr_bitmap.m_width * exr_bitmap.m_height * exr_bitmap.m_channels);
        return exr_bitmap;
    }
    exr_bitmap.m_width      = width;
    exr_bitmap.m_height     = height;
    exr_bitmap.m_channels   = 4;
    exr_bitmap.m_image_data = std::vector<float>(data, data + (width * height * 4));

    free(data);  // release memory of loaded image data

    return exr_bitmap;
}

inline bool write_png(
    const char *filename, unsigned width, unsigned height, unsigned channels, const unsigned char *pixels)
{
    assert(channels == 4);
    return stbi_write_png(filename, width, height, channels, pixels, width * sizeof(uint32_t));
}

inline bool write_exr(const char *filename,
                      size_t width,
                      size_t height,
                      size_t channels,
                      const float *pixels,
                      float rgb_renormalisation  = 1.f,
                      ExrCompression compression = ExrCompression::EXR_COMPRESSION_NONE)
{
    constexpr size_t num_channels = 4;
    if (width == 0 || height == 0 || channels != num_channels || !pixels) {
        logger(LogLevel::Error, "Invalid image passed to write_exr\n");
        return false;
    }

    const size_t num_pixels = width * height;

    std::string path{filename};

    // EXR expects the channels to be separated. We also apply a potential renormalisation (e.g. due to
    // accumulated samples)
    std::vector<float> separated(num_channels * num_pixels);
    for (size_t c = 0; c < num_channels; ++c) {
        float *tgt = separated.data() + c * num_pixels;
        for (size_t i = 0; i < num_pixels; ++i) {
            // Flip for ABGR order!
            tgt[i] = pixels[num_channels * i + c];
            // if (c != 3)
            tgt[i] /= rgb_renormalisation;
        }
    }

    float *img_ptrs[num_channels];
    img_ptrs[0] = separated.data() + 3 * num_pixels;  // Flip order!
    img_ptrs[1] = separated.data() + 2 * num_pixels;
    img_ptrs[2] = separated.data() + 1 * num_pixels;
    img_ptrs[3] = separated.data();

    // Note: Channels must be in alphabetical order.
    EXRChannelInfo channel_info[num_channels];
    memset(channel_info, 0, sizeof(EXRChannelInfo) * num_channels);

    channel_info[0].name[0] = 'A';
    channel_info[1].name[0] = 'B';
    channel_info[2].name[0] = 'G';
    channel_info[3].name[0] = 'R';

    int pixel_types[] = {
        TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT};

    EXRHeader header;
    InitEXRHeader(&header);
    header.num_channels          = num_channels;
    header.channels              = channel_info;
    header.pixel_types           = pixel_types;
    header.requested_pixel_types = pixel_types;

    // Compression is slow; >1s for a ZIP compressed full hd image, or 0.5 s
    // for PIZ compressed. So if possible this should be done in a post process.
    switch (compression) {
    case EXR_COMPRESSION_ZIP:
        header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
        break;
    case EXR_COMPRESSION_PIZ:
        header.compression_type = TINYEXR_COMPRESSIONTYPE_PIZ;
        break;
    case EXR_COMPRESSION_RLE:
        header.compression_type = TINYEXR_COMPRESSIONTYPE_RLE;
        break;
    default:
        header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;
        break;
    }

    EXRImage image;
    InitEXRImage(&image);
    image.num_channels = num_channels;
    image.images       = reinterpret_cast<unsigned char **>(img_ptrs);
    image.width        = (int)width;
    image.height       = (int)height;

    const int ret = SaveEXRImageToFile(&image, &header, path.c_str(), nullptr);
    if (ret != TINYEXR_SUCCESS) {
        logger(LogLevel::Error, "Failed to write %s\n", path.c_str());
    }
    return ret == TINYEXR_SUCCESS;
}

}