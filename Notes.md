For the GeoDelta Plot 

Plot the plans first and get the bounding box from those to set the map extent then plot the 
outlines and clip

Also make an "add_labels" function that adds labels to internal points of polygons


Okay, I think that I want to go back to Layers as their own class

make these internal and use them
that should make things nicer for devs and we can hide the implementation details
from the users

continuous_color_layer
discrete_color_layer

outline and highlight layers can just be special calls to the discrete color layer

move seats-votes computation to scoring module


For Census

- Add in a way to get shapefiles
- Better name converter for census data