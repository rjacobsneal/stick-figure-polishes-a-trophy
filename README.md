# Stick Figure Polishes a Trophy

10-second animation rendered using drawing in c++ (final project for CPSC 478: Computer Graphics)

LINK: https://youtu.be/WepprhhpGE4
-----------

FEATURES:
-----------
I implemented a glossy reflection in the sphere. 
    Each ray that hits the sphere generates 100 new, unqiue sample rays.
    Each sample ray is reflected with a direction that has been adjusted by a randomized direction.
    The randomized directions all are within the hemisphere about the ray's perfect reflection.
    The sphere also is shaded using standard shading models as explored in the last homework assignment. 
I implemented textures in the floor, metal box, and windows on wall. 
    The program reads a .ppm file and sets up a Tecture object. 
    When an intersection with a triangle is detected, the triangle is given barycentric coordinates. 
    These coordinates are used to map and then sample the pixel color from a texture image.

HOW TO RUN:
-----------
Make the previz executable with a standard "make" call
    "./previz" will begin to render all 300 frames
    "./previz 1" will render a single representative frame (frame 52)
