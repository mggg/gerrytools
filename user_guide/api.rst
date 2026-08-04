API reference
=============

The API reference is organized by the public module from which users import an object. The user
guides explain workflows and tradeoffs; these pages provide signatures, parameter contracts,
return types, and the detailed behavior recorded in docstrings.

Core workflow modules
---------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Data
      :link: api/data
      :link-type: doc

      Census, ACS, and CVAP retrieval plus processed 2020 geographic downloads.

   .. grid-item-card:: Plan comparison
      :link: api/plan_comparison
      :link-type: doc

      Population and area overlap, dispersion, and optimal district relabeling.

   .. grid-item-card:: Scoring
      :link: api/scoring
      :link-type: doc

      Scoring-engine plan evaluation and array-based derived scores.

   .. grid-item-card:: Plotting
      :link: api/plotting
      :link-type: doc

      Plot builders, option dataclasses, and Matplotlib style objects.

   .. grid-item-card:: MGRP
      :link: api/mgrp
      :link-type: doc

      Docker-backed ReCom, Forest, and SMC runners with constraints and objectives.

   .. grid-item-card:: BEN
      :link: api/ben
      :link-type: doc

      Record GerryChain runs as self-describing BENDL ensemble files.

   .. grid-item-card:: LaTeX
      :link: api/latex
      :link-type: doc

      TeX documents, tables, formatters, and TeX-native plot builders.

Supporting modules
------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Colors
      :link: api/colors
      :link-type: doc

      Named palettes, color resolution, and districtr color utilities.

.. toctree::
   :hidden:
   :maxdepth: 1

   Data <api/data>
   Plan comparison <api/plan_comparison>
   Scoring <api/scoring>
   Plotting <api/plotting>
   MGRP <api/mgrp>
   BEN <api/ben>
   LaTeX <api/latex>
   Colors <api/colors>
