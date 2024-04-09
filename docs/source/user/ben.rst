.. _ben:

BEN (Binary-Ensamble)
=====================

The ``ben`` module is a simple Docker wrapper that allows the user to run 
versions of the 
`binary-ensamble <https://github.com/peterrrock2/binary-ensemble>`_, 
`msms_parser <https://github.com/peterrrock2/msms_parser>`_,
and `smc_parser <https://github.com/peterrrock2/smc_parser>`_ cli tools.


If the user has cargo installed on their system and is comfortable with
using cli tools, then it is generally recommended recommended that they
use the cli tools directly. However, for anyone that is not comfortable
using the terminal, cannot install cargo, or would like a single workflow
file for running various items in ``gerrytools``, we have provided this
convenience module.

.. admonition:: For Jupyter Notebook Users
    :class: warning

    Many of the tools in this module will print a progress string to the terminal
    to let the user know how far along the compression, decompression, or parsing
    process is. However, there are instances (mostly in the parsing methods) where 
    the program executes so fast that the Jupyter Client is overwhelmed by the output
    and will stall (this will likely cause Jupyter to prompt you to restart the
    kernel). You do not need to restart the kernel, but it would be a good idea to
    set the ``verbose`` flag in each method to ``False`` to prevent this from
    happening.


Compression
-----------

The main workhorse for the compression tools within the `ben` module come from
the `binary-ensamble <https://github.com/peterrrock2/binary-ensemble>`_ cli tool.
For more information on how the compression algorithm works and how to use the
cli tool directly, please refer to the above link. 

.. raw:: html 

    <div class="center-container">
        <a href="https://raw.githubusercontent.com/peterrrock2/binary-ensemble/main/example/small_example.jsonl", class="download-badge", download>
        small_example.jsonl
        </a>
    </div>
    <br style="line-height: 5px;"> 


The compression and decompression part of this package are primarily handled by the
:func:`ben` function. With the exception of the ``xz-compress`` and ``xz-decompress``
modes, which serve as general compression utilities for any file type, the main
modes of the :func:`ben` function are made to work with the standard JSONL format
of the ``mgrp`` module:

.. code::

    {"assignment": <assignment_vector>, "sample": <sample_number_indexed_from_1>}



which can be run in several different ways. First, make sure that you
have the ``ben`` module imported:

.. code:: python

    from gerrychain.ben import *

- ``encode`` This mode will convert a JSONL file to a BEN file:

.. code:: python

    ben(
        mode="encode",
        input_file_path="./small_example.jsonl",
    )

- ``x-encode``

.. code:: python

    ben(
        mode="x-encode",
        input_file_path="./small_example.jsonl.ben",
    )


- ``decode``

.. code:: python

    ben(
        mode="decode",
        input_file_path="./small_example.jsonl.ben",
        output_file_path="./re_small_example.jsonl",
    )

- ``x-decode``

.. code:: python

    ben(
        mode="x-decode",
        input_file_path="./small_example.jsonl.xben",
        output_file_path="./re_small_example_v2.jsonl",
    )

- ``xz-compress``

.. code:: python 
    
    ben(
        mode="xz-compress",
        input_file_path="./small_example.jsonl",
        output_file_path="./compressed_small_example.jsonl.xz",
    )

- ``xz-decompress``

.. code:: python

    ben(
        mode="xz-decompress",
        input_file_path="./compressed_small_example.jsonl.xz",
        output_file_path="./decompressed_small_example.jsonl",
    )
