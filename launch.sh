docker build -t gpgpu .
docker run -it --gpus all gpgpu /bin/bash
