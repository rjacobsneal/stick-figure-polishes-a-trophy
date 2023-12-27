# Stick Figure Polishes a Trophy
[Like to YouTube video](https://youtu.be/WepprhhpGE4)

## Description:
Final project for CPSC 478: Computer Graphics: advanced 10-second animation rendered using C++ graphics

## Langauges:
C++

## Features:
### Glossy Reflection in the Sphere
- Each ray that hits the sphere generates 100 new, unique sample rays.
- Each sample ray is reflected with a direction that has been adjusted by a randomized direction.
- The randomized directions are all within the hemisphere about the ray's perfect reflection.
- The sphere is shaded using standard shading models as explored in the last homework assignment.

### Textures
- Implemented textures in the floor, metal box, and windows on the wall.
- The program reads a .ppm file and sets up a Texture object.
- When an intersection with a triangle is detected, the triangle is given barycentric coordinates.
- These coordinates are used to map and then sample the pixel color from a texture image.

## How to Run:
Make the `previz` executable with a standard "make" call.
- `./previz` will begin to render all 300 frames.
- `./previz 1` will render a single representative frame (frame 52).
