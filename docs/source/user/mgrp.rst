MGRP (Metric Geometry Replication)
==================================

The main purpose of the ``mgrp`` module is to provide the user with a simple,
intuitive, and ergonomic way to run and replicate some of the most common
methods used to generate ensembles of districting plans.

The ``mgrp`` module requires that the user has a working installation of
:ref:`Docker <docker_setup>` on their machine. The module will also need
access to the internet the first time that a function is run since it needs
to download the 
`necessary Docker image <https://hub.docker.com/repository/docker/mgggdev/replicate/general>`_
to run the functions inside of the appropriate containerized environment. In the event
that the user would like to instead build the Docker image locally, they can
then reference
`this repository <https://github.com/peterrrock2/mgggdev-replicate-docker-info>`_
for more information.

The ``mgrp`` module comes with three major modes:

1. **Recom** (frcw): This mode allows the user to make use of the 
   `Rust implementation <https://github.com/mggg/frcw.rs>`_ of our 
   `gerrychain <https://gerrychain.readthedocs.io/en/latest/>`_ package.
   The ``mgrp`` module will still allow for the use of custom updaters, but
   things like custom constraints are not currently supproted. 
   
   Since the ``frcw`` code is written in Rust, it is significantly faster than
   the Python implementation. This means that the user can run things like
   region aware chains on the block level in a reasonable amount of time
   compared to what would be required in the Python implementation (for
   reference, a 100k chain on the block level could take about 6 months to
   run in the Python implementation, and in Rust it takes less than 12 hours).


2. **ForestRecom** (MSMS): This mode allows the user to make use of what has been
   termed the "Forest Recombination" method. This method was originally put forth by 
   a team at Duke and was 
   `written in the Julia <https://git.math.duke.edu/gitlab/quantifyinggerrymandering/>`_.
   The interested user can consult the original paper 
   `Multi-Scale Merge-Split Markov Chain Monte Carlo for Redistricting <https://arxiv.org/pdf/2008.08054.pdf>`_

3. **Sequential Monte Carlo** (SMC): This mode allows the user to make use of the 
   `Sequential Monte Carlo <https://github.com/alarm-redist/redist>`_ method. This 
   method was originally put forth by a team at Harvard and was written using a 
   combination of C++ and R. For more information on the project, please consult 
   `the project's main page <https://alarm-redist.org/redist/>`_.


All of these methods will make use of the ``RunContainer`` class which handles all of the
interaction with the Docker container. Here are the basic attributes of the ``RunContainer``
class:

- ``configuration``: This specifies the configuration that should be used to run the chain.
  and is set by either the ``RecomRunnerConfig``, ``ForestRunnerConfig``, or ``SMCRunnerConfig``
  classes.
- ``docker_image_name``: The name of the docker image that you would like to use to run the chain.
  This does not generally need to be set since it is set to the default value of 
  ``mgggdev/replicate:v0.2`` which contains all of the necessary dependencies to run the
  ``mgrp`` module.
- ``docker_client_args``: This is only used for debugging purposes in the event that you have
  multiple docker contexts on the same machine. The normal user does not need to concern themselves
  with this setting.



Recom (frcw)
------------

The basic usage for the ``Recom`` mode relies on the following two classes:


``RecomRunnerConfig``: This class is used to help configure the ``RunContainer`` class
which will be talking to the Docker container that is running the ``frcw`` code. 

- ``json_file_path``: The path to the dual graph JSON file that we would like to
  run the chain on.
- ``output_folder``: The directory where the output of the chain should be written. This 
  will default to "./output" if not specified.
- ``log_folder``: The directory where the logs of the chain should be written. This will
  default to "./logs" if not specified.


``RecomRunInfo``: This class contains all of the information that is needed to
run a chan using the ``frcw`` code. The structure of this class reflects the
arguments that can be passed to the main cli used in ``frcw``. 

- ``pop_col``: The name of the column in the graph that contains the population
  data.
- ``assigment_col``: The name of the column in the graph that contains the
  the district assignment data.
- ``variant``: The name of the variant that should be used to run the chain. The
  options for this are as follows:

    - ``A`` This uses the cut-edges minimum spanning tree method. So from the 
      set of cut edges, one is selected at random and the two districts lying
      on either side of the cut edge are selected for recombination. The resulting
      districts are then merged together, and a minimal spanning tree is constructed
      using Kruskal's algorithm. The tree is then split into two new districts
      by finding a balanced cut within the minimum spanning tree.
    - ``B`` This uses a district-pairs minimum spanning tree method. So from the
      set of adjacent district pairs, one is selected at random and the two
      districts are then merged together and a minimal spanning tree is constructed
      using Kruskal's algorithm. The tree is then split into two new districts
      by finding a balanced cut within the minimum spanning tree.
    - ``C`` This uses the cut-edges uniform spanning tree method. Everything is the
      same as in variant A, except that the minimal spanning tree is constructed
      using Wilson's algorithm.
    - ``D`` This uses the district-pairs uniform spanning tree method. Everything is the
      same as in variant B, except that the minimal spanning tree is constructed
      using Wilson's algorithm.
    - ``R`` This is the reversible recombination method. For more information on the
      the inner workings of this method, please consult the paper 
      `Spanning Tree Methods for Sampling Graph Partitions <https://arxiv.org/pdf/2210.01401.pdf>`_
    - ``AW`` This is the cut-edges region aware method. This method is similar to the
      cut-edges method, but it also allows the user to specify certain regions that they
      would like to try and keep together. Edges between different regions are then given
      a higher weight according to the surcharges that may be specified in the `region weights`
      parameter, and the minimal spanning tree is constructed using Kruskal's algorithm.
    - ``BW`` This is the district-pairs region aware method. This method is similar to the
      district-pairs method, but it also allows the user to specify certain regions that they
      would like to try and keep together. Edges between different regions are then given
      a higher weight according to the surcharges that may be specified in the `region weights`
      parameter, and the minimal spanning tree is constructed using Kruskal's algorithm.
- ``balance_ub``: The upper bound on the number of balance edges to be used in the ``R`` variant.
- ``n_steps``: The number of steps that the chain should run for. Note: with the way that the
  ``frcw`` code is currently written, the output of the chain will likely overshoot this by
  a few steps due to any self loops not being accounted for in the count until the next successful
  proposed plan is found.
- ``pop_tol``: The allowable population deviation from ideal when drawing districts.
- ``n_threads``: The number of cpu-threads that should be used to run the chain. This is especially
  useful when running a chain that might experience a lot of self-looping / rejection.
- ``batch_size``: The number of allowable steps per unit of multi-threaded work. Again, this is
  especially useful when running a chain that might experience a lot of self-looping / rejection.
- ``writer`` The type of writer that should be used to write the output of the chain. The options
  for this are as follows:

    - ``canonical`` This is the default option, and writes the output of the chain to a JSONL
      file using the format ``{"assignment": <assignment-vector>, "sample": <sample-number>}``.
      This is the standard format used across all of the methods in the ``mgrp`` module.
    - ``ben`` This writes the output assignment vectors using the BEN compression algorithm.
      To learn more, please see the :ref:`ben module of this package <binary-ensamble>`.
    - ``json`` This will write the output of the chain to a json file. The assignment vectors
      are not recorded in this mode, but statistical information about the chain such as which
      districts were merged, the number of self loops, and the tallies of relevant statistics
      are recorded.
    - ``jsonl-full`` Like the json writer, but this will also record the assignment vectors.
    - ``pcompress`` This will write the assignment vectors of the chain to a file compressed
      using the `PCompress <https://github.com/mggg/pcompress>`_ compression algorithm (a 
      delta encoding algorithm).
    - ``assignments`` This will write the assignment vectors of the chain to a file prefixed
      with the step number.
    - ``canonicalized-assignments`` This will write the assignment vectors of the chain to a file
      prefixed with the step number, but the assignment vectors will be canonicalized in the sense
      that the assignment vectors will be renumbered to start at 1, so [3,3,1,1,4,4,2,2] would
      become [1,1,2,2,3,3,4,4].
- ``rng_seed``: The seed that should be used to initialize the random number generator.
- ``region_weights``: This is a dictionary that contains the region weights that should be used
  in the ``AW`` and ``BW`` variants. The keys of the dictionary should be the region names, and
  the values should be the surcharge that should be applied to the edges between the regions.
- ``force_print``: This is a boolean that determines whether or not the output of the chain
  should be printed to the console. This can be useful for debugging purposes
- ``updaters``: This a dictionary of updaters that can be used in conjunction with the
  ``mcmc_run_with_updaters`` method of the ``RunContainer`` class.

An Example of Running a Chain Using the ``Recom`` Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html 

    <div class="center-container">
        <a href="https://github.com/peterrrock2/gerrytools-dev/blob/main/tutorials/data/50x50.json", class="download-badge", download>
        50x50 Dual Graph
        </a>
    </div>
    <br style="line-height: 5px;"> 


As always, the first thing that we need to do is import the necessary modules:

.. code:: python

    from gerrytools.mgrp import *

Then we can set up the configuration and run info classes:

.. code:: python

    recom_config = RecomRunnerConfig(
        json_file_path="./50x50.json",
    )

    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="district",
        variant="A",
        n_steps=1000,
        rng_seed=123456,
    )

and now we set up the ``RunContainer`` class so that it can run the chain:

.. code:: python

    with RunContainer(recom_config) as c:
        c.run(run_info)


of course, we might want to use some custom updaters in our chain, so we can do that as well:

.. code:: python

    from gerrychain import Graph, Partition

    def cut_edge_count(partition):
        graph = partition.graph
        cut_edges = 0
        for edge in graph.edges:
            if partition.assignment[edge[0]] != partition.assignment[edge[1]]:
                cut_edges += 1
        return cut_edges


    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="district",
        variant="A",
        n_steps=11,
        updaters={"my_cut_edges": cut_edge_count},
        rng_seed=42
    )


Since we have updaters, we need to make sure to iterate over the output of the chain
using the ``mcmc_run_with_updaters`` method:

.. code:: python

    with RunContainer(recom_config) as c:
        for output, error in c.mcmc_run_with_updaters(run_info):
            if output is not None:
                print(output)

Forest Recom (MSMS)
-------------------


``ForestRunnerConfig``: This class is used to help configure the ``RunContainer`` class
which will be talking to the Docker container that is running the MSMS code. 

- ``json_file_path``: The path to the dual graph JSON file that we would like to
  run the chain on.
- ``output_folder``: The directory where the output of the chain should be written. This 
  will default to "./output" if not specified.
- ``log_folder``: The directory where the logs of the chain should be written. This will
  default to "./logs" if not specified.


``ForestRunInfo``: This class contains all of the information that is needed to
run a chan using the MSMS code. The structure of this class reflects the
arguments that can be passed to the main cli that we have defined for the MSMS code.
This cli fundamentally calls the ``run_metropolis_hastings`` under the hood. For more
information on the cli we use here, please see 
`this link <https://github.com/peterrrock2/mgggdev-replicate-docker-info/tree/main/home/forest/cli>`_

- ``region_name``: The name of the greater region that we would like to use to help us split into
  districts.
- ``subgregion_name``: The name of the subregion that we would like to use to help us split into
  districts.
- ``pop_col``: The name of the column in the graph that contains the population information.
- ``num_dists``: The number of districts that we would like to split the graph into.
- ``pop_dev``: The allowable population deviation from ideal when drawing districts.
- ``gamma``: The gamma parameter given in the MSMS paper. This parameter should be between
  0 and 1, and when it is 0, the chain will sample uniformly from the space of possible
  spanning forests. When it is 1, the chain will sample uniformly from the space of possible
  partitions.
- ``n_steps``: The number of steps that the chain should run for.
- ``rng_seed``: The seed that should be used to initialize the random number generator.
- ``output_file_name``: The name of the file that the output of the chain should be written to.
- ``standard_jsonl``: A boolean that determines whether or not the output of the chain should
  be written in the standard JSONL format 
  ``{"assignment": <assignment-vector>, "sample": <sample-number>}``. For consistency, with
  the rest of the outputs in the ``mgrp`` module, this is set to True by default.
- ``ben``: A boolean that determines whether or not the output of the chain should be written
  using the BEN compression algorithm. For more information on this, please see the 
  :ref:`ben module of this package <binary-ensamble>`.
- ``force_print``: This is a boolean that determines whether or not the output of the chain
  should be printed to the console. This can be useful for debugging purposes
- ``updaters``: This a dictionary of updaters that can be used in conjunction with the
  ``mcmc_run_with_updaters`` method of the ``RunContainer`` class.


.. warning::

    If the ``standard_jsonl`` and ``ben`` flags are both set to False, then the output format
    of the MSMS method will be exceptionally large and will likely take up a lot of space on
    the user's machine. It is recommended that the user only set these flags to False if they
    are sure that they have enough space on their machine to store the output.

    In the event that the user has some MSMS output that they would like to then convert to
    the standard JSONL format, or to the BEN format, they can make use of the
    :func:`~gerrytools.ben.msms_parse` function.

An Example of Running a Chain Using the ``Forest`` Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. raw:: html 

    <div class="center-container">
        <a href="https://github.com/peterrrock2/gerrytools-dev/blob/main/tutorials/data/50x50.json", class="download-badge", download>
        NC Dual Graph
        </a>
    </div>
    <br style="line-height: 5px;"> 


Once more we import the necessary modules:

.. code:: python

    from gerrytools.mgrp import *

Then we can set up the configuration and run info classes:

.. code:: python

    forest_config = ForestRunnerConfig(
        json_file_path="./NC_pct21.json",
    )

    run_info = ForestRunInfo(
        region_name="county",
        subregion_name="prec_id",
        pop_col="pop2020cen",
        num_dists=14,
        pop_dev=0.01,
        gamma=0,
        n_steps=33,
        rng_seed=123456,
    )

and now we set up the ``RunContainer`` class so that it can run the chain:

.. code:: python

    with RunContainer(forest_config) as c:
        c.run(run_info)


Fortunately for us, ForestRecom is also a MCMC method, so we can also use custom updaters
while we run it!

.. code:: python

    from gerrychain import Graph, Partition

    def cut_edge_count(partition):
        graph = partition.graph
        cut_edges = 0
        for edge in graph.edges:
            if partition.assignment[edge[0]] != partition.assignment[edge[1]]:
                cut_edges += 1
        return cut_edges


    run_info = ForestRunInfo(
        region_name="county",
        subregion_name="prec_id",
        pop_col="pop2020cen",
        num_dists=14,
        pop_dev=0.01,
        gamma=0,
        n_steps=33,
        updaters={"my_cut_edges": cut_edge_count},
        rng_seed=42
    )



Since we have updaters, we need to make sure to iterate over the output of the chain
using the ``mcmc_run_with_updaters`` method:

.. code:: python

    with RunContainer(recom_config) as c:
        for output, error in c.mcmc_run_with_updaters(run_info):
            if output is not None:
                print(output)


Sequential Monte Carlo (SMC)
----------------------------

The SMC mode is a bit different from the other two modes in that it is not an MCMC method
and in that there are actually 3 different classes that need to be appropriately set up
in order to run the ensemble. The number of toggles on these classes are quite substantial,
as well, but the user can consult the `main documentation <https://alarm-redist.org/redist/>`_
for more information on the toggles that are available. 

``SMCRunnerConfig``: This class is used to help configure the ``RunContainer`` class

- ``shapefile_dir``: The directory that contains the shapefile.
- ``shapefile_name``: The name of the shapefile that should be used in the SMC algorithm.
- ``output_folder``: The directory where the output files should be written to. Defaults to "./output".
- ``log_folder``: The directory where the log files should be written to. Defaults to "./logs".


``SMCMapInfo``: This class contains all of the information needed to construct the 
`redist_map() <https://alarm-redist.org/redist/reference/redist_map.html>` object
that is used in the R code.

- ``pop_col``: The name of the column in the shapefile that contains the population data.
- ``n_dists``: The number of districts that the shapefile should be split into.
- ``pop_tol``: The allowable population deviation from ideal when drawing districts.
- ``pop_bounds``: Optional custom population bounds to be use in the ``redist_map()`` function.
  This needs to be a list of three ints: [lower_bound, target, upper_bound].


``SMCRedistInfo``: This class contains all of the information needed to construct the 
  `redist_smc() <https://alarm-redist.org/redist/reference/redist_smc.html>` object that is used
  in the R code. We have chosen to preserve the default values for these parameters that
  were set by the ALARM team.

- ``n_sims``: Teh number of samples that should be drawn to form the ensemble.
- ``rng_seed``: The seed that should be used to initialize the random number generator.
  Defaults to 42.
- ``compactness``: Controls the compactness of the generated districts. Defaults to 1.0.
- ``resample``: A boolean that determines whether to perform a final resampling step so that the generated plans can be used immediately. Defaults to False.
- ``adapt_k_thresh``: The threshold value used in the heuristic to select a value of :math:`k_i` for each splitting iteration. Must be in the range [0, 1]. Defaults to 0.985
- ``seq_alpha``: Determines the amount to adjust the weights by at each resampling step. 
  Must be in the range [0, 1]. Defaults to 0.5.
- ``pop_temper``: Controls the strength of the automatic population tempering. Defaults to
  0.0, but if the algorithm is having trouble then it is recommended to start looking at
  values in the range 0.01-0.05.
- ``final_infl``: A multiplier for the population constraint on the final iteration. Used to 
  loosen the constraint when the sampler is getting suck on the final split. Defaults to 1.0.
- ``est_label_mult``: A multiplier for the number of importance samples to use in estimating 
  the number of ways to sequentially label the districts. Defaults to 1.0.
- ``verbose``: A boolean that determines whether or not to log the intermediate information 
  during the running of SMC. This is suppressed by the JSONL and BEN outputs generally.
  Defaults to False.
- ``silent``: A boolean that determines whether or not to suppress all diagnostic output.
  Defaults to False.
- ``tally_columns``: A list of columns to be tallied into the output file. This is only 
  generated if the ``standard_jsonl`` and ``ben`` flags are set to False.
- ``output_file_name``: The desired name of the output file. If not set, then the file name 
  will be determied according to a set of heuristics. Not set by default.
- ``standard_jsonl``: A boolean that determines whether or not the output of the chain should
  be written in the standard JSONL format 
  ``{"assignment": <assignment-vector>, "sample": <sample-number>}``.
- ``ben``: A boolean that determines whether or not the output of the chain should be written
  using the BEN compression algorithm. For more information on this, please see the
  :ref:`ben module of this package <binary-ensamble>`. Defaults to False.



An Example of Running an Ensemble Using the ``SMC`` Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. raw:: html 

    <div class="center-container">
        <a href="https://github.com/peterrrock2/gerrytools-dev/blob/main/tutorials/data/4x4.zip", class="download-badge", download>
        4x4 Shapefile
        </a>
    </div>
    <br style="line-height: 5px;"> 

We know the drill by now, we import the necessary modules, set up the information

.. code:: python

    from gerrytools.mgrp import *

    smc_config = SMCRunnerConfig(
        shapefile_dir="./",
        shapefile_name="4x4_grid",
    )

    map_info = SMCMapInfo(pop_col="TOTPOP", n_dists=4)

    redist_info = SMCRedistInfo(
        n_sims=29,
        tally_columns=["TOTPOP"],
        verbose=True,
    )


and run the container:

.. code:: python

    with RunContainer(smc_config) as c:
        c.run(
            map_info = map_info,
            redist_info = redist_info
        )