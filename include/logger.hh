#pragma once

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <optional>
#pragma once

#include <string>

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
void init_logging(const bool verbose,
                  const std::optional<std::string>& log_file); 

/**
 * Change the logging level
 * @param level The logging level
 */
void set_logging_level(const std::string& level);
