#include "logger.hh"

/**
 * Initialize the logging system
 * @param verbose Whether to enable verbose logging to the console
 * @param log_file The path to the log file
 */
void init_logging(const bool verbose,
                  const std::optional<std::string>& log_file)
{
    // Set up the logger for the console
    if (verbose)
    {
        logging::add_console_log(std::cout,
                                 keywords::format =
                                     (expr::stream << "["
                                                   << logging::trivial::severity
                                                   << "] " << expr::smessage));
    }

    // Set up the logger for the file
    if (log_file)
    {
        logging::add_file_log(keywords::file_name = *log_file,
                              keywords::format =
                                  (expr::stream
                                   << "[" << logging::trivial::severity << "] "
                                   << expr::smessage));
    }

    logging::core::get()->set_filter(logging::trivial::severity
                                     >= logging::trivial::info);

    logging::add_common_attributes();
}

/**
 * Change the logging level
 * @param level The logging level
 */
void set_logging_level(const std::string& level)
{
    trivial::severity_level l = trivial::info;

    switch (level[0])
    {
    case 't':
        l = trivial::trace;
        break;
    case 'd':
        l = trivial::debug;
        break;
    case 'i':
        l = trivial::info;
        break;
    case 'w':
        l = trivial::warning;
        break;
    case 'e':
        l = trivial::error;
        break;
    case 'f':
        l = trivial::fatal;
        break;
    }

    logging::core::get()->set_filter(logging::trivial::severity >= l);
}