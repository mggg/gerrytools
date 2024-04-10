.. _ben:

BEN (Binary-Ensemble)
=====================

The ``ben`` module is a simple Docker wrapper that allows the user to run 
versions of the 
`binary-ensemble <https://github.com/peterrrock2/binary-ensemble>`_, 
`msms_parser <https://github.com/peterrrock2/msms_parser>`_,
and `smc_parser <https://github.com/peterrrock2/smc_parser>`_ CLI tools.


If the user has cargo installed on their system and is comfortable with
using CLI tools, then it is generally recommended that they
use the CLI tools directly. However, for anyone that is not comfortable
using the terminal, cannot install cargo, or would like a single workflow
file for running various items in ``gerrytools``, we have provided this
module for your convenience.


.. admonition:: Make Your Docker Desktop Client is Current!
    :class: warning

    The ``ben`` module uses Docker to run the CLI tools. If you have not
    updated your Docker Desktop Client in a while, it is recommended that
    you do so before running any of the methods in this module.

    Specifically, if you are running a client that is older than version 4.28.0,
    then you will want to update since there have been significant improvements
    in the way that VirtioFS and Rosetta work on the newer versions of Docker Desktop.

    If you are running MacOS or Windows without WSL2 (Windows Subsystem for Linux),
    then you will need to make sure that you have VirtioFS enabled in the Docker
    Desktop Client. This can be found in the settings wheel under the "General"
    tab. (The other opeions are gRPC Fuze and osxfs and are **not** recommended for
    use since they significantly hamper file i/o). If you do not see this option
    and are on > 4.28.0, then you do not need to worry.


.. admonition:: For Jupyter Notebook Users
    :class: tip

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

The main workhorse for the compression tools within the ``ben`` module come from
the `binary-ensemble <https://github.com/peterrrock2/binary-ensemble>`_ CLI tool.
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

    from gerrytools.ben import *

- ``encode`` This mode will convert a JSONL file to a BEN file:

.. code:: python

    ben(
        mode="encode",
        input_file_path="./small_example.jsonl",
    )

- ``x-encode`` This mode can be used to convert either a JSONL or BEN file to an
  XBEN file:

.. code:: python 

    ben(
        mode="x-encode",
        input_file_path="./small_example.jsonl.ben",
    )


- ``decode`` This mode can be used to convert an XBEN file to a BEN file or a BEN
  file to a JSONL file:

.. code:: python

    ben(
        mode="decode",
        input_file_path="./small_example.jsonl.ben",
        output_file_path="./re_small_example.jsonl",
    )

- ``x-decode`` This mode can be used to convert an XBEN file to a JSONL file:

.. code:: python

    ben(
        mode="x-decode",
        input_file_path="./small_example.jsonl.xben",
        output_file_path="./re_small_example_v2.jsonl",
    )

- ``xz-compress`` This mode can be used as a general compression utility for any
  file type:

.. code:: python 
    
    ben(
        mode="xz-compress",
        input_file_path="./small_example.jsonl",
        output_file_path="./compressed_small_example.jsonl.xz",
    )

- ``xz-decompress`` This mode can be used as a general decompression utility for any
  file that was compressed with the ``xz-compress`` mode (or with level 9 xz compression): 

.. code:: python

    ben(
        mode="xz-decompress",
        input_file_path="./compressed_small_example.jsonl.xz",
        output_file_path="./decompressed_small_example.jsonl",
    )

Improving Compression Via Relabeling
------------------------------------

Underneath the hood, the BEN algorithm uses some simple run-length encoding (RLE)
followed by bit-packing to compress our data. So if we have a simple assignment
vector like:

.. code::

    [1,1,1,2,2,2,2,3,1,3,3,3]

the BEN algorithm will encode this as:

.. code::

    [(1,3), (2,4), (3,1), (1,1), (3,3)]

which is then bit-packed to the following

.. code::

    01011101
    00110010
    10011101
    10000000

It is not important exactly how this is all done at the moment, but the interested
reader by refer to the documentation of the 
`binary-ensemble <https://github.com/peterrrock2/binary-ensemble>`_
CLI tool for more information.

This turns a list that previously took ~48 bytes to store (if we exclude the commas and the
brackets) into something that takes ~4 bytes. So, in order to make the compression better,
we would prefer the nodes in the assignment vector to be ordered in such a way that
adjacent nodes are more likely to be assigned to the same district since this will
shorten the run-length encoding (observe that if we resort the above assignment vector,
we can get an rle of ``[(1,4),(2,4),(3,4)]`` which fits into 2 bytes). 


.. raw:: html 

    <div class="center-container">
        <a href="https://raw.githubusercontent.com/peterrrock2/binary-ensemble/main/example/CO_small.json", class="download-badge", download>
        CO Dual Graph
        </a>
        <a href="https://raw.githubusercontent.com/peterrrock2/binary-ensemble/main/example/100k_CO_chain.jsonl", class="download-badge", download>
        CO Ensemble
        </a>
    </div>
    <br style="line-height: 5px;"> 

We will be making use of the above CO Dual Graph and CO Ensemble files to demonstrate how much
we can improve the compression by relabeling the nodes in the assignment vector.

First thing is first, we need to extract the XBEN file into a BEN file. This will take up ~7Gb, 
but make sure that you don't extract it to a JSONL file since the JSONL file will be ~27Gb.

.. code:: python

    ben(
        mode="decode",
        input_file_path="100k_CO_chain.xben"
    )


This should take ~5min to complete. **If this takes longer than 10 min, then you need to
check that your Docker Desktop Client is up to date and that VirtioFS is enabled.**

The very first thing that we can do to improve the compression is to canonicalize the
assignment vectors. Why does this help? This is best explained by example. Consider the
following assignment vectors:

.. code::

    [2,2,3,3,1,1,4,4]
    [2,2,3,3,4,4,1,1]

We, as humans, can see that these are describing the same partition of the districts,
but our computer lacks the relevant context to make this connection, so we need to
help it along a little bit:

.. code:: python

    canonicalize_ben_file(
        input_file_path="100k_CO_chain.jsonl.ben"
    )


This will produce the file ``100k_CO_chain_canonicalized_assignments.jsonl.ben`` which
will (counterintuitively) be a little larger than the original file by a small margin.
However, if you were to compress this file using XBEN, you would find the new
``100k_CO_chain_canonicalized_assignments.jsonl.xben`` file to be around 1/3 the 
size of our starting file (DON'T actually do this since it will take over an hour).

The next thing that we would like to do is to decide on a good labeling order to use for
the nodes in the graph. In general, there will not be a *best* ordering to use, but
since we are trying to partition a state, sorting by some geographic information like
GEOID is generally a good place to start.

.. code:: python

    relabel_json_file_by_key(
        dual_graph_path="CO_small.json",
        key="GEOID20",
        # uncomment the next line if you are running this in a Jupyter Notebook
        # verbose=False 
    )

This command will produce a new "map" file that will contain the information that we need
to do the relabeling. This is then accomplished by running the following command:

.. code:: python

    relabel_ben_file_with_map(
        input_file_path="100k_CO_chain_canonicalized_assignments.jsonl.ben",
        map_file_path="CO_small_sorted_by_GEOID20_map.json"
    )

This will produce a new file called 
``100k_CO_chain_canonicalized_assignments_sorted_by_GEOID20.jsonl.ben``
and you should find that this file is ~550Mb -- almost exactly the same size as the
XBEN file that we downloaded at the start of this! But, we can do even better than this
by using the ``x-encode`` mode to convert this file to an XBEN file:

.. code:: python

    ben(
        mode="x-encode",
        input_file_path="100k_CO_chain_canonicalized_assignments_sorted_by_GEOID20.jsonl.ben"
    )

This will produce an XBEN file that is practically microscopic compared to the original -- ~6Mb!
Of course, with the exception of the canonicalization step, we have also made sure to record
all of the transformations that we have made to the data so that we can reverse them at any time,
and we have the added benefit of being able to send what used to be a 27Gb file to someone else
in an email.


.. tip::

    The above two-step relabeling process can actually be accomplished with a 
    single command by using the ``relabel_ben_file_by_key`` method:

    .. code:: python

        relabel_ben_file_by_key(
            input_file_path="100k_CO_chain_canonicalized_assignments.jsonl.ben",
            dual_graph_path="CO_small.json",
            key="GEOID20",
            # uncomment the next line if you are running this in a Jupyter Notebook
            # verbose=False
        )

Parsing MSMS and SMC Output
---------------------------

.. raw:: html 

    <div class="center-container">
        <a href="https://raw.githubusercontent.com/peterrrock2/binary-ensemble/main/example/msms_out.zip", class="download-badge", download>
        Forest Output
        </a>
        <a href="https://raw.githubusercontent.com/peterrrock2/gerrytools-dev/main/tutorials/data/smc_out.zip", class="download-badge", download>
        SMC Output
        </a>
    </div>
    <br style="line-height: 5px;"> 


In some situations it may be desirable to turn an alternative output of the
Forest Recom or Sequential Monte Carlo (SMC) algorithms into a JSONL or a 
BEN file. This will be less common given the default settings in ``mgrp``,
but it is still good to know how to do this.

Let us start with the Forest Recom output. The native julia output of the Forest
Recom code tends to be exceedlingly (for example, a 1M chain on PA [9255 nodes] 
will be over 200Gb). So, it is sometimes necessary to convert this output to something
more manageable.