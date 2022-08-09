# FEA2TexturedModel

![topology_optimized fea texture](https://github.com/ddm-j/FEA2TexturedModel/blob/master/texturedModelRender.png)

Converts a 3D mesh file (STL, OBJ, etc) and a corresponding scalar point map file (stress, strain, pressure, vorticity, etc) into a single colormapped, textured 3D model in OBJ/MTL format. 

## Useage

### For Stationary Image Texture: 
This will generate a naive UV unwrapped texture that is created from interpolating vertex colors onto the faces of a 3D mesh file. The texture will be output with a corresponding OBJ and MTL model for 3D display and embedding in pressentations, web, etc.

`fea_texture.py path/to/3Dfile.stl path/to/pointmap.csv --cmap colormapname --smooth`

The CLI accepts any colormap name that is supported by `matplotlib`. The default colormap is "turbo" although this is not industry standard for displaying FEM data. Additionally, the `--smooth` option can be passed, which uses a laplacian smoothing algorithm to even out the color texture where the mapping/interpolation may have created bad looking regions.

Example with provided files:
`fea_texture.py data/exampleMesh.stl data/exampleFEA.csv --cmap viridis`
Notably, here I'm passing the "viridis" colormap for the texture instead of using the default "turbo".

### For Video Image Texture: 
This will create an MP4 video texture. This is not supported by default in the OBJ/MTL file format. But by using a 3D animation package (Blender, C4D, ...), one may use this video texture to animate simulation data on the corresponding 3D model.

`fea_texture_animated.py path/to/3Dfile.stl path/to/timesteps stepCount --cmap colormapname --length 6 --fps 60`

Here we specify a 3D file, the path to the folder containing pointmap data of name `pointmap#.csv` where # is the number of the timestep, the number of corresponding timesteps, the matplotlib colormap name, the length of the video file, and the framerate of the video file. 

Example with provided files:
`fea_texture_animated.py data/exampleMesh.stl data/animation --cmap summer --length 3 --fps 120`
Notably, here I'm passing the "viridis" colormap for the texture instead of using the default "turbo".
