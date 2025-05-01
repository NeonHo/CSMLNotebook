 the command line:

nvcc --version
(or /usr/local/cuda/bin/nvcc --version) gives the CUDA compiler version (which matches the toolkit version).

From application code, you can query the runtime API version with

cudaRuntimeGetVersion()
or the driver API version with

cudaDriverGetVersion()
As Daniel points out, deviceQuery is an SDK sample app that queries the above, along with device capabilities.

As others note, you can also check the contents of the version.txt (.json in recent CUDA versions) using (e.g., on Mac or Linux)

cat /usr/local/cuda/version.txt
However, if there is another version of the CUDA toolkit installed other than the one symlinked from /usr/local/cuda, this may report an inaccurate version if another version is earlier in your PATH than the above, so use with caution.