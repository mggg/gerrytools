# TODO

- [x] Update the developer workflow to use a makefile and uv

- [ ] Ben module
  - [x] Improve integration with binary-ensemble
  - [ ] Add some python bindings for the MSMS parser and the SMC parser
    - [ ] In the SMC parser and runner, maybe just save in the RDS format and then do the
          conversion on top of that.
    - [ ] In MSMS, allow for arbitrary numbers of levels to be parsed

- [ ] Data Module
  - [ ] Update the ACS5 module to use the new Census API
  - [ ] Try to remove some of the dependencies of this module
  - [ ] Add some fallback mirrors for the data (e.g. Chicago's mirrors)
  - [ ] Change the way that we estimate CVAP (WolfRam?)

- [ ] Geometry Module
  - [x] Improve the way that we calculate dispersion. Move over to scipy implementation.
  - [x] Remove the dissolve function
  - [ ] Remove the invert function
  - [x] Update `minimize_dispersion` and `minimize_dispersion_with_parity`
  - [x] Remove unit map (done in Maup now)

- [x] MGRP Module
  - [x] Fix api reference to actually grab all of the functions for this
  - [x] Update the docker image for all of the methods

- [x] Plotting Module
  - [x] Make examples showing how all of the outputs look in this
  - [x] Make some nice interfaces for the plots we needed for UT
  - [x] Change `arrow` to `add_arrow`
  - [x] Make a colormaps module (some with districtr colors, some with latex colors, etc.)
    - [x] Add some of the Matplotlib and seaborn colormaps into this module for easy use
    - [x] Making the names constants would also be good.
  - [ ] Figure out what "gif_multidimentional" is doing
  - [x] Change `ideal` to `add_vertical_lines` and then add many of them. Include ability to
        jitter the lines if they might overlap
  - [x] Update scatterplot so that you can add labels to the points and allow for sorting
        the values
  - [x] Improve the interface for violin plots

- [ ] Scoring Module
  - [ ] Support a void district label in `RegionParts` so incomplete plans exclude unassigned
        nodes (districtr)
  - [ ] Return part counts by region label instead of only the aggregate `RegionParts` total
        (districtr)
  - [ ] Support preparing a child-only evaluator from a hybrid parent/child graph without copying
        the child graph, for persistent service evaluators (districtr)
