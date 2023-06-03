#pragma once

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <optional>

namespace logging = boost::log;
namespace keywords = boost::log::keywords;
namespace sinks = boost::log::sinks;
namespace expr = boost::log::expressions;
namespace attrs = boost::log::attributes;
namespace src = boost::log::sources;
namespace trivial = boost::log::trivial;

/**
 * Initialize the logging system
 * @param verbose Whether to enable verbose logging to the console
 * @param log_file The path to the log file
 */
void init_logging(bool verbose, std::optional<std::string> log_file) {
  // Set up the logger for the console
  if (verbose) {
    logging::add_console_log(
        std::cout,
        keywords::format = (expr::stream << "[" << logging::trivial::severity
                                         << "] " << expr::smessage));
  }

  // Set up the logger for the file
  if (log_file) {
    logging::add_file_log(keywords::file_name = *log_file,
                          keywords::format =
                              (expr::stream << "[" << logging::trivial::severity
                                            << "] " << expr::smessage));
  }

  logging::core::get()->set_filter(logging::trivial::severity >=
                                   logging::trivial::info);

  logging::add_common_attributes();
}

/**
 * Change the logging level
 * @param level The logging level
 */
void set_logging_level(trivial::severity_level level) {
  logging::core::get()->set_filter(logging::trivial::severity >= level);
}
