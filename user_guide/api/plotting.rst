:tocdepth: 3

Plotting
========

For task-oriented examples, see the :doc:`plotting guides <../user/plotting/index>`.

Plot builders and top-level options
-----------------------------------

.. automodule:: gerrytools.plotting
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Unset

Matplotlib axis and title styles
--------------------------------

These public style objects live in :mod:`gerrytools.plotting.mpl` rather than the top-level
plotting namespace.

.. autoclass:: gerrytools.plotting.mpl.AxisLabelStyle
   :members:

.. autoclass:: gerrytools.plotting.mpl.TitleStyle
   :members:

.. autoclass:: gerrytools.plotting.mpl.TickStyle
   :members:

Label style presets
-------------------

The registry behind the ``LabelOptions(style=...)`` argument on the map builders; each entry
is a :class:`~gerrytools.plotting.LabelStyle`. See the
:doc:`geographic plot controls guide <../user/plotting/geographic/options>` for the rendered
gallery.

.. autodata:: gerrytools.plotting.mpl.label_text_options.LABEL_STYLES
   :no-value:

Font type aliases
-----------------

.. autodata:: gerrytools.plotting.mpl.FontFamily

.. autodata:: gerrytools.plotting.mpl.FontStretch

.. autodata:: gerrytools.plotting.mpl.FontStyle

.. autodata:: gerrytools.plotting.mpl.FontVariant

.. autodata:: gerrytools.plotting.mpl.FontWeight
