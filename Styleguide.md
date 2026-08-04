# Gerrytools Style Guide


## Latex

This module is used for generating LaTeX code and for generating previews of LaTeX code.
Since the preview portion of this module will function similarly to the Plotting module in
the mind of users, the naming and design conventions should be similar to those used in the 
Plotting module.

## Plotting

This module is made as a wrapper on matplotlib, so an effort should be made to name variables
in a way that is consistent with matplotlib's naming conventions. For the most part, this just
means that parameters corresponding to visual elements should be named using flatcase 
(e.g. `line2d`, `axhline`, etc.) rather than snake_case (e.g. `line_2d`, `ax_hline`, etc.). 
Functions should still be named using snake_case, as per Python conventions and classes should
be named using PascalCase.
