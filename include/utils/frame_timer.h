#pragma once

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "utils/logger.h"

namespace ntwr {

class FrameTimer {
public:
    FrameTimer() {}

    void start_frame(uint32_t frame_idx)
    {
        if (frame_idx == 0) {
            frame_times.clear();
        }
        start_time = std::chrono::high_resolution_clock::now();
    }

    void end_frame()
    {
        const auto current_time                                        = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> frame_duration = current_time - start_time;
        start_time                                                     = current_time;
        frame_times.push_back(frame_duration.count());
    }

    void write_to_file(std::string filename, std::vector<float> *losses = nullptr)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            logger(LogLevel::Error, "Error opening file: %s", filename.c_str());
            return;
        }

        if (losses) {
            file << "frame index,frame time (ms), loss\n";
        } else {
            file << "frame index,frame time (ms)\n";
        }

        for (size_t frame_idx = 0; frame_idx < frame_times.size(); frame_idx++) {
            if (losses) {
                file << frame_idx << "," << frame_times[frame_idx] << "," << (*losses)[frame_idx] << "\n";
            } else {
                file << frame_idx << "," << frame_times[frame_idx] << "\n";
            }
        }

        file.close();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::vector<double> frame_times;
};

}