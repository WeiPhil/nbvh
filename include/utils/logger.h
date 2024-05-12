#pragma once

#include <atomic>
#include <iostream>
#include <mutex>
#include "tinyformat.h"

namespace ntwr {

/// @brief The severity of a log message.
enum LogLevel {
    Debug = 0,
    Info  = 1,
    Warn  = 2,
    Error = 3,
};

/// @brief The interface used to log messages to console output.
class Logger {
    /// @brief Synchronization to ensure that messages from different threads
    /// are not intermangled.
    std::mutex m_mutex;
    /// @brief A status message to be shown at the bottom of console output
    /// (e.g., render progress in percent).
    std::string m_status;

public:
    /// @brief Logs a message to console output, which will be constructed from
    /// the given format string and corresponding arguments.
    template <typename... Args>
    void operator()(LogLevel level, const char *fmt, const Args &...args)
    {
        std::unique_lock lock{m_mutex};

        std::cout << "\033[2K\r";

        switch (level) {
        case LogLevel::Debug:
            std::cout << "\033[90m[Debug] \033[0m";
            break;
        case LogLevel::Info:
            std::cout << "\033[32m[Info] \033[0m";
            break;
        case LogLevel::Warn:
            std::cout << "\033[33m[Warn] \033[0m";
            break;
        case LogLevel::Error:
            std::cout << "\033[31m[Error] \033[0m";
            break;
        }
        (level >= LogLevel::Error ? std::cerr : std::cout) << tfm::format(fmt, args...) << std::endl;

        std::cout << m_status << std::flush;
    }

    /// @brief Sets the status text for display at the bottom of console output,
    /// constructed from a given format string.
    template <typename... Args>
    void setStatus(const char *fmt, const Args &...args)
    {
        std::unique_lock lock{m_mutex};
        m_status = tfm::format(fmt, args...);
        std::cout << "\033[2K\r" << m_status << std::flush;
    }
};

/// @brief The interface used to log messages to console output.
extern Logger logger;

}
