#include <boost/program_options.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "frame-helper.cuh"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "segmentation-helper.cuh"

using namespace cv;
namespace po = boost::program_options;

// TODO : support more options
int main(int argc, char** argv)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "input,i", po::value<std::string>(), "set the input video path")(
        "output,o", po::value<std::string>(), "set the output video path");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    if (!vm.count("input") || !vm.count("output"))
    {
        std::cout << "Both input and output video files must be specified.\n";
        return 1;
    }

    std::string input_path = vm["input"].as<std::string>();
    std::string output_path = vm["output"].as<std::string>();

    process_frames(input_path, output_path);

    return 0;
}
