# Custom Layers for Tensorflow

## SE3 Composition Layer

### Dependency

- Sophus : https://github.com/strasdat/Sophus
  - Need to clone the repository, build, and install.

### Build

If you use python3, please edit 'CMakeLists.txt' file before you build.

```
$ cd deepvo/networks/layers
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```