# N-BVH: Neural ray queries with bounding volume hierarchies

![Alt text](./scene_data/teaser.jpg "N-BVH Teaser")

Source code of the paper "N-BVH: Neural ray queries with bounding volume hierarchies" by [Philippe Weier](https://weiphil.github.io/portfolio/), Alexander Rath, Élie Michel, Iliyan Georgiev, Philipp Slusallek, Tamy Boubekeur at SIGGRAPH 2024.

You can find the original paper and supplemental viewer on the [project webpage](https://weiphil.github.io/portfolio/neural_bvh).

## Building the project

We tested our implementation with the following configuration:

Windows:
- Visual Studio 17 2022
- Cuda Compiler NVIDIA 12.0
- Nvidia Driver 546.29
- Cuda Architectures 86/89
- RTX 3080 / RTX 4090

Linux:
- Ubuntu 22.04
- Cuda Compiler NVIDIA 12.3
- Nvidia Driver 545.29
- Clang++ 14.0
- RTX 3080 / RTX 4090

### Cloning the repository

Make sure to clone the repository with all the submodules:

```bash
git clone --recursive git@github.com:WeiPhil/nbvh.git
```

#### Required Cuda Compute Capability

Our implementation depends on the FullyFusedMLP implementation of TinyCudaNN which requires at least Compute Capability 7.5.
Make sure you set the variable CMAKE_CUDA_ARCHITECTURES="your_gpu_compute_capabability" appropriately for your GPU. A list with the compute capability associated with every GPU can be found [here](https://developer.nvidia.com/cuda-gpus)

### Windows

From the nbvh project directory configure the project as follows (replace the cuda architecture with yours):

```bash
cmake -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=89 -A x64 -B build
```
And build and run in Release mode using:
```bash
cmake --build build --config Release && .\build\Release\ntwr.exe
```

### Linux

From the nbvh project directory configure the project as follows (replace the cuda architecture with yours):

```bash
cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -B build
```
And build and run in Release mode using:
```bash
cmake --build build --config Release && ./build/ntwr
```
 
## Getting the scenes used in the paper

The repository only ships with the Chess scene (`nbvh\scenes_path_tracing\chess\chess.gltf`). The other scenes used in the paper can be downloaded [from the University of Saarland Cloud Server](https://oc.cs.uni-saarland.de/owncloud/index.php/s/b33TTcX9ZCS2m2X). The folder contains:
- Environment maps that are used across the different scenes (`hdris`) 
- Scenes designed for the hybrid neural path tracing module (`scenes_path_tracing`)
- Scenes used for the neural prefiltering module (`scenes_prefiltering`)

Our renderer supports the [GLTF](https://www.khronos.org/gltf/) scene format (`.gltf/.glb`), converting to this scene format can be easilly done in [Blender](https://www.blender.org/).

## Usage

### Opening a scene

To open a scene, simply drag and drop a `.gltf` file in the viewer. Note that this might take a moment for larger scenes.

Alternatively you can also use the menu bar `File->Load Scene` to select a scene on your system.

### Loading a configuration file

Every scene we provide for download comes with a `configs` directory which contains the different network/nbvh configurations we used in the paper. You can load one of the configuration file directly through the menu bar `File->Load Config->config_file.json`. Loading a config file will also load any associated pre-trained nbvh and network weights if available in the `network_models` and `nbvhs` directory respectively. To reduce the download size, we only provide the pre-trained weights and nbvhs that corresponds to the results displayed in the main comparison figure of our paper (Figure 6 and 10).

### Running inference and/or optimisation

Once a configuration file has been loaded you can start the optimisation process (if pre-trained data isn't available) by clicking the `Run optimisation` checkbox in the `Neural BVH Renderer` window. Inference can also be run simultaneously (at the cost of a lower optimisation framerate) by clicking `Run inference`.

### Rendering the reference

The (non-neural) reference of a scene can be rendered by clicking `Disable Neural BLAS` in the `Neural BVH Renderer` window and clicking `Run inference`. This simply replaces all the N-BVH BLAS's with their original non-neural counterparts while rendering.

### Further details for neural prefiltering of scenes

Scenes used for neural prefiltering depend on pre-trained prefilted appearance from our paper ["Neural Prefiltering for Correlation-Aware Levels of Detail"](https://github.com/WeiPhil/neural_lod). Please follow the instructions there to generate pre-trained appearance for your own scene. You also need to explicitely set the `Neural BVH Module` used to `Neural Prefiltering` in the `Neural BVH Renderer` window if you don't run one of our given configuration file. When the N-BVH module is switched, the renderer will automatically look for pre-trained prefilted appearance in the scene's directory (see one of our provided prefiltered scene directory structure). 

## Creating your own N-BVH compatible scene

To render your own scene with our pipeline you will need to create your scene in Blender and export it to the GLTF format. While you will be able to open, optimise and render the scene as expected, our renderer will assume the entire scene is neural. To indicate wether an object should be rendered neurally or not we use two different [GLTF scenes](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_004_ScenesNodes.html), one for the entire scene and one containing only the neural parts of the scene. The easiest way to create the expected GLTF format is to create an additional scene node in Blender. 

The process in Blender is as follows:
- Create your scene in Blender as usual
- Create another scene node in Blender with name `scene_neural` (our renderer looks for the _neural suffix) [(see Blender manual)](https://docs.blender.org/manual/en/latest/scene_layout/scene/introduction.html)
- For every object in the initial scene that should be neural, in Blender, select `Object->Link/Transfer Data->Link Objects to Scene->scene_neural`

The created scene can then be exported as usual as GLTF file and loaded into the renderer. An example Blender file for the Chess scene is provided in `nbvh\scenes_path_tracing\chess\chess.blend`. 

## Controlling reconstruction quality, performance and memory

The default configuration when opening a scene is to learn a tree cut that results in a N-BVH of about 2k nodes. While this might be enough for small scenes, very large scenes can easilly require more than 150k nodes for good quality. 

The target number of nodes in the N-BVH is best controlled via the `Split scaling` factor in the `BVH Split Scheduler` collapsing header. Note that increasing the number of nodes is mostly going to impact the performance rather than the memory footprint. Memory footprint is impacted most by the hashmap size (and number of features per level). The hashmap configuration can be found in the `Input Encoding` tab. Increasing the hashmap size can also increase quality, however, at a much smaller scale than increasing the node count. 

We recomend first increasing the node count before tweaking the hashmap size as the later might drastically increase the memory consumption of the model. Check out our [Interactive Viewer](https://weiphil.github.io/portfolio/neural_bvh_viewer/) tp get more intuition on the different configuration tradeoffs.

### Citation

```bibtex
@inproceedings{Weier:2024:NeuralBVH,
  author = {Philippe Weier and Alexander Rath and \'{E}lie Michel and Iliyan Georgiev and Philipp Slusallek and Tamy Boubekeur},
  title = {N-BVH: Neural ray queries with bounding volume hierarchies},
  booktitle = {ACM SIGGRAPH 2024 Conference Proceedings},
  year = {2024},
  doi = {10.1145/3641519.3657464},
  isbn = {979-8-4007-0525-0/24/07}
}
```