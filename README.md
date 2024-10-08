# IRGPUA PROJECT

## Project presentation :

### Introduction :

AMDIA company has a mission for you :</br>
The internet is broken → All images are corrupted and are unusable

After some deep analysis performed by the LRE labs, the issue was found :</br>
A hacker named Riri has introduced garbage data at the entry of Internet pipelines

The corruption is as follows :
* Garbage data in the form of lines of “-27” have been introduced in the image</br>
* Pixels got their values modified : 0 : +1 ; 1 : -5 ; 2 : +3 ; 3 : -8 ; 4 : +1 ; 5 : -5; 6 : +3; 7 : -8…</br>
* All images have awful colors and should be histogram equalized

Engineers from AMDIA have found a solution but it is super slow (CPU folk’s issues)</br>
Your goal is to fix the issue in real-time

AMDIA knows you will succeed in the mission and has a final task for you :</br>
Once all images from the pipeline are cleaned up, they want to do some statistics</br>
You should first compute each image’s total sum of all its pixel values</br>
Then sort the resulting images based on this total sum (optional)</br>
Of course, all of this should be done in the fastest way possible

They will provide you with their CPU code to start and test your implementation

### Advices :

Almost all of the algorithms / patterns used for the project will be practiced in TP</br>
Do the hands-on and ask questions to have the best base to start from</br>
Implementing a working (aka slow) version of the remaining algorithms / patterns should not be too complicated</br>
Once and only once your pipeline is working, you should start optimizing</br>
You will be working in groups of 3. Split tasks (parallel work) and communicate to share ideas about optimization ideas. You should think together but I advise you to code separately so that you all learn and have time to finish the project.</br>
If you have any questions : nblin@epita.fr

### Expectations :

* Have a working pipeline (fix images + statistics; sorting on GPU is optional)
* Have optimized algorithms / patterns, memory allocations…
* Have Nsight Compute & Systems reports (screenshots) to show your performance analysis
* Have a second working industrial (fast) version using CUB, Thrust for the algorithms (you should minimize the use of hand-made kernel as much as possible for this version)
* Have a presentation (or report, as you wish) explaining how you programmed, analyzed, and optimized each algorithm / pattern and how you used libraries for your industrial version
* I prefer having algorithms extra optimized rather than great slides ! Don’t spend too much time on it
* Project Code and Slides (or report) needs to be sent before 04/11/2024 - 23h42

### Mininal expectations :

* Have a working pipeline (fix images + statistics)
* Have a working industrial version with no handwritten kernel
* Have the full final Reduce introduced in class and programmed during hands-on
* Have a working Scan (block + grid level)
* Have a histogram at least as good as last one introduced in class
* Use some of the introduced programming tools
* Have at least one Nsight Systems overall reports analysis


### Maximal expectations :

* All previous
* Performance / Benchmark Analysis with Nsight Compute of the different Reduce steps
* Different block-wide Scan + Decouple Lookback + Nsight Compute Analysis
* Optimize further all other patterns with introduced techniques
* Use all the introduced programming tools
* Deep Nsight Systems analysis
* More pattern/overall optimizations that were not introduced in this class ;)

If you are up for the challenge :
* GPU Radix Sort

If you are crazy :
* GPU Decouple Lookback Radix Sort

### About the code

Path for the images has been hardcoded for you and should be working on the OpenStack.

If you still want to download the "images" folder : https://cloud.lrde.epita.fr/s/XYPimokPGSQrM35
(Feel free to then change the "images" path in main.cu if you have your downloaded the folder)

The project layout is simple :
* An "images" folder containing the broken images you need to fix (it is in the afs)
* "main.cu" contains the code used to : Load the images from the pipeline, fix the images on CPU & compute the statistics and writing the results
* "fix_cpu.cu/h" contains the code that fixes images on CPU
* "image.hh" and "pipeline.hh" simply contains the code to handle both images and pipeline

I advise you to work with only one image at first to avoid file loading time. Moreover it will be easier to debug if you always work with the same image.

You can modify all the code as you wish but I have highlighted the code with "TODO" where you should act.
You should not have to modify the pipeline.h. You can but make sure the baseline CPU code still works **exactly** the same.
You **must not** modify the fix_cpu files.
You **must change** the main.cu to fix images on both CPU & GPU and check your results.

Feel free to create more files, classes...

PS : I know some images look weird after the CPU histogram equalization, it is "normal"
(To be fair, not all the images are actually good candidates for the histogram equalization
But for the exercise, having you do an optimized histogram is cool :)

### Requirements to build

* [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
* C++ compiler ([g++](https://gcc.gnu.org/) for linux,  [MSVC](https://visualstudio.microsoft.com/downloads/) for Windows)
* [GPU supported by CUDA](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
* [CMake](https://cmake.org/download/)

### Build

!! If you are not on the OpenStack, I strongly advise you to remove the first lines in the CmakeLists.txt !!

- To build, execute the following commands :

```bash
mkdir build && cd build
cmake ..
make -j
```

* By default the program **will run in release**. To build in **debug**, do:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Run :

```bash
cd build
./main
```
