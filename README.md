# gpgpu

![Build linux](https://github.com/downvoteed/GPGPU/actions/workflows/build-linux.yml/badge.svg)

## Requirements

- CMake
- Ninja
- OpenCV
- Boost

## Build

```
cmake -S . -B build
cmake --build build
```

## Run CPU

### Help

```sh
./build/src/cpu/cpu -h
```

### Must use options

- `-v` or `--verbose` : verbose mode
- `-j` or `--jobs` : number of threads (default: 1)
- `-i` or `--input` : input video file path
- `-o` or `--output` : output video file path
- `-f` or `--fps` : fps of the video (default: 24)
- `-d` or `--display` : display video (0: no, 1: yes, default: 1)
- `-w` or `--webcam` : use webcam as input
- `--width` : width of the output video
- `--height` : height of the output video
- `--background-optimizer` : background optimizer (default for webcam)

## Run GPU

### Must use options

- `-i` or `--input` : input video file path
- `-o` or `--output` : output video file path


### Example

```sh
./build/src/cpu/cpu -i ./dataset/video.avi
./build/src/cpu/cpu -w
```

## Known issues

- Output video has blinking frames on RTX 4060.
