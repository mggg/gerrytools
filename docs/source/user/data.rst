====
Data
====

The ``data`` module of ``gerrytools`` is designed to handle the loading and the
processing of data. In partucular, it provides methods for loading and managing
data from the census. Below is a basic tutorial on what we veiw to be a common
workflow for someone using this module.


.. note::

    Sometimes, when calling functions that work with the 
    `us <https://pypi.org/project/us/>`_ 
    module (and ``data`` does do this), you may see the following error:

    .. code-block:: console

        ValueError: Unexpected response (URL: ...): Sorry, the system is currently
        undergoing maintenance or is busy. Please try again later.
    
    This is due to a Census API issue and cannot be fixed from the python side of
    things. However, re-running the code generally fixes the issue.


For this tutorial, we will assume that the reader is working through with
a Jupyter notebook. All of the required packages needed to run this tutorial
can be found in the 
`tutorial\_requirements.txt <../_static/tutorial_requirements.txt>`_
file.


Census
------

There are two different methods that ``data`` provides for loading census data:
:meth:`~gerrytools.data.census20` and :meth:`~gerrytools.data.census10` . As the
names would suggest, the former is liked to the US census data collected in 2020, and
the latter is linked to the US census data collected in 2010. There is significant
difference between the two methods, so please be sure to refer to the documentation.

For the purposes of this tutorial, we will be using the 2020 census data. The first
thing to do is to load all of the necessary packages:

.. code-block:: python

    from gerrytools.data import *
    import geopandas as gpd
    import pandas as pd
    import us

And now we would like to load the census data for the state of Massachusetts. When we
go to load this data, we should be aware that there are 5 different tables that are 
available on the Census Bureau's API for retrieving the 2020 Decennial Census 
PL 94-171 at the stated level of geography. These tables are:

- **P1**: Race
- **P2**: Hispanic or Latino, and Not Hispanic or Latino by Rac
- **P3**: Race for the Population 18 Years and Over (Race by VAP)
- **P4**: Hispanic or Latino, and Not Hispanic or Latiny by Race for the Population
  18 Years and Over
- **P5**: Group Quarters Population by Group Quarters Type

.. code-block:: python

    df = census20(
        us.states.MA,
        table="P3",
        columns={},
        geometry="tract",    
    )

    df[["GEOID20", "VAP20", "WHITEVAP20", "BLACKVAP20", "ASIANVAP20", "OTHVAP20"]].head()

In jupyter, this will display the following table:

+---+------------+------+-----------+-----------+-----------+---------+
|   |  GEOID20   | VAP20| WHITEVAP20| BLACKVAP20| ASIANVAP20| OTHVAP20|
+===+============+======+===========+===========+===========+=========+
| 0 | 25001012601| 2657 | 1868      | 153       | 122       | 172     |
+---+------------+------+-----------+-----------+-----------+---------+
| 1 | 25001012602| 4564 | 2444      | 517       | 147       | 547     |
+---+------------+------+-----------+-----------+-----------+---------+
| 2 | 25001012700| 4059 | 3445      | 119       | 49        | 144     |
+---+------------+------+-----------+-----------+-----------+---------+
| 3 | 25001012800| 3464 | 2971      | 86        | 42        | 84      |
+---+------------+------+-----------+-----------+-----------+---------+
| 4 | 25001012900| 3568 | 3011      | 101       | 47        | 103     |
+---+------------+------+-----------+-----------+-----------+---------+

Of course, anyone that is familiar with the way that the census data is
organized would realize that the column names here not the same as the
ones that the Census Bureau uses. This is because the ``census20`` method
has it's own mapping of the Census Bureau's column names to the ones that
are a bit easier to understand. If you would like to see the mapping, you
use the :meth:`~gerrytools.data.variables` method; so, for the "P3" table,
we would call:

.. code-block:: python

    variables("P3") 

Which outputs the following:

.. code-block:: console 

    {'P3_001N': 'VAP20',
    'P3_003N': 'WHITEVAP20',
    'P3_004N': 'BLACKVAP20',
    'P3_005N': 'AMINVAP20',
    'P3_006N': 'ASIANVAP20',
    'P3_007N': 'NHPIVAP20',
    'P3_008N': 'OTHVAP20',
    'P3_011N': 'WHITEBLACKVAP20',
    'P3_012N': 'WHITEAMINVAP20',
    'P3_013N': 'WHITEASIANVAP20',
    'P3_014N': 'WHITENHPIVAP20',
    'P3_015N': 'WHITEOTHVAP20',
    'P3_016N': 'BLACKAMINVAP20',
    'P3_017N': 'BLACKASIANVAP20',
    'P3_018N': 'BLACKNHPIVAP20',
    'P3_019N': 'BLACKOTHVAP20',
    'P3_020N': 'AMINASIANVAP20',
    'P3_021N': 'AMINNHPIVAP20',
    'P3_022N': 'AMINOTHVAP20',
    'P3_023N': 'ASIANNHPIVAP20',
    'P3_024N': 'ASIANOTHVAP20',
    'P3_025N': 'NHPIOTHVAP20',
    'P3_027N': 'WHITEBLACKAMINVAP20',
    'P3_028N': 'WHITEBLACKASIANVAP20',
    'P3_029N': 'WHITEBLACKNHPIVAP20',
    'P3_030N': 'WHITEBLACKOTHVAP20',
    'P3_031N': 'WHITEAMINASIANVAP20',
    'P3_032N': 'WHITEAMINNHPIVAP20',
    'P3_033N': 'WHITEAMINOTHVAP20',
    'P3_034N': 'WHITEASIANNHPIVAP20',
    'P3_035N': 'WHITEASIANOTHVAP20',
    'P3_036N': 'WHITENHPIOTHVAP20',
    'P3_037N': 'BLACKAMINASIANVAP20',
    'P3_038N': 'BLACKAMINNHPIVAP20',
    'P3_039N': 'BLACKAMINOTHVAP20',
    'P3_040N': 'BLACKASIANNHPIVAP20',
    'P3_041N': 'BLACKASIANOTHVAP20',
    'P3_042N': 'BLACKNHPIOTHVAP20',
    'P3_043N': 'AMINASIANNHPIVAP20',
    'P3_044N': 'AMINASIANOTHVAP20',
    'P3_045N': 'AMINNHPIOTHVAP20',
    'P3_046N': 'ASIANNHPIOTHVAP20',
    'P3_048N': 'WHITEBLACKAMINASIANVAP20',
    'P3_049N': 'WHITEBLACKAMINNHPIVAP20',
    'P3_050N': 'WHITEBLACKAMINOTHVAP20',
    'P3_051N': 'WHITEBLACKASIANNHPIVAP20',
    'P3_052N': 'WHITEBLACKASIANOTHVAP20',
    'P3_053N': 'WHITEBLACKNHPIOTHVAP20',
    'P3_054N': 'WHITEAMINASIANNHPIVAP20',
    'P3_055N': 'WHITEAMINASIANOTHVAP20',
    'P3_056N': 'WHITEAMINNHPIOTHVAP20',
    'P3_057N': 'WHITEASIANNHPIOTHVAP20',
    'P3_058N': 'BLACKAMINASIANNHPIVAP20',
    'P3_059N': 'BLACKAMINASIANOTHVAP20',
    'P3_060N': 'BLACKAMINNHPIOTHVAP20',
    'P3_061N': 'BLACKASIANNHPIOTHVAP20',
    'P3_062N': 'AMINASIANNHPIOTHVAP20',
    'P3_064N': 'WHITEBLACKAMINASIANNHPIVAP20',
    'P3_065N': 'WHITEBLACKAMINASIANOTHVAP20',
    'P3_066N': 'WHITEBLACKAMINNHPIOTHVAP20',
    'P3_067N': 'WHITEBLACKASIANNHPIOTHVAP20',
    'P3_068N': 'WHITEAMINASIANNHPIOTHVAP20',
    'P3_069N': 'BLACKAMINASIANNHPIOTHVAP20',
    'P3_071N': 'WHITEBLACKAMINASIANNHPIOTHVAP20'}

For more information on the variables that are available in each of these
tables, please refer to the 
`census website <https://api.census.gov/data/2020/dec/pl/variables.html>`_ . 

ACS5
----

This is a method that is used to load the 5-year American Community Survey
data that that he Census Bureau uses for the 5-year population estimates
of the United States. 

.. warning:: 

    The ACS5 data uses geometries from the 2010 census, and not the 
    2020 census. 

.. code-block:: python

    acs5_df = acs5(
        us.states.MA,
        geometry="block group", # data granularity, either "tract" (default) or "block group"
        year=2019, 
    )
    acs5_df[["BLOCKGROUP10", "TOTPOP19", "WHITE19", "BLACK19", "ASIAN19", "OTH19"]].head()

This will print the following table: 

+---+---------------+---------+--------+--------+--------+------+
| i | BLOCKGROUP10  | TOTPOP19| WHITE19| BLACK19| ASIAN19| OTH19|
+===+===============+=========+========+========+========+======+
| 0 | 250173173012  | 571     | 340    | 15     | 137    | 0    |
+---+---------------+---------+--------+--------+--------+------+
| 1 | 250173531012  | 1270    | 660    | 311    | 93     | 0    |
+---+---------------+---------+--------+--------+--------+------+
| 2 | 250173222002  | 2605    | 2315   | 61     | 96     | 21   |
+---+---------------+---------+--------+--------+--------+------+
| 3 | 250251101035  | 1655    | 1077   | 242    | 82     | 0    |
+---+---------------+---------+--------+--------+--------+------+
| 4 | 250251101032  | 659     | 158    | 225    | 0      | 0    |
+---+---------------+---------+--------+--------+--------+------+

Estimating CVAP
---------------

.. raw:: html 

    <div class="center-container">
        <a href="https://github.com/peterrrock2/gerrytools-dev/blob/main/docs/source/_static/al_bg.zip", class="download-badge", download>
        AL Block Group Shapefile
        </a>
    </div>
    <br style="line-height: 5px;">

Sometimes, we might want to estimate the citizen voting age population (CVAP)
for a particular demographic group. This is especially true in the case where we are
working with potentially new geometries for a particular state, as tends to happen
after the Decennial census, which we would like to use to make projections based on
our previous knowledge of the state demographics. In our case, we will be using the
:meth:`~gerrytools.data.estimate_cvap10` method to estimate the CVAP for particular
geometries in the year 2020 using information from the previous ACS.

The :meth:`~gerrytools.data.estimate_cvap10` method wraps the above ``cvap()`` and ``acs5()`` 
functions to help users pull forward CVAP estimates from 2019 (on 2010 geometries) to
estimates for 2020 (on 2020 geometries). To use this, one must supply a base
geodataframe with the 2020 geometries on which they want CVAP estimates. Additionally, users
must specify the demographic groups whose CVAP statistics are to be estimated. For
each group, users specify a triple :math:`(X, Y, Z)` where :math:`X` is the old CVAP column for
that group, :math:`Y` is the old VAP column for that group, and :math:`Z` is the new VAP column
for that group, which must be an existing column on ``base``.  Then, the estimated new
CVAP for that group will be constructed by multiplying :math:`X / Y \cdot Z` for each new
geometry.

Let's start with grabbing the geometries for Alabama and looking at the ``acs5()`` 
and ``cvap()`` data:

.. code-block:: python

    base = gpd.read_file("al_bg")
    acs5_cvap19 = acs4(us.states.AL, year=2019)
    cvap_cvap19 = cvap(us.states.AL, year=2019)



.. admonition:: Tips for picking :math:`X`, :math:`Y`, and :math:`Z`
    :class: tip

    Your :math:`X` should be any CVAP column returned by either ``acs5()`` or ``cvap()``,
    so anything generated by:

    .. code-block:: python

        print([col for col in pd.concat([acs_cvap19, cvap_cvap19]) if "CVAP" in col])])

    Which, in our case, would be:

    .. code-block:: console

        ['WHITECVAP19', 'BLACKCVAP19', 'AMINCVAP19', 'ASIANCVAP19', 'NHPICVAP19', 'OTHCVAP19, 
        '2MORECVAP19', 'NHWHITECVAP19', 'HCVAP19', 'CVAP19', 'POCVAP19', 'CVAP19e', 'NHCVAP19', 
        'NHCVAP19e', 'NHAMINCVAP19', 'NHAMINCVAP19e', 'NHASIANCVAP19', 'NHASIANCVAP19e', 
        'NHBLACKCVAP19', 'NHBLACKCVAP19e', 'NHNHPICVAP19', 'NHNHPICVAP19e', 'NHWHITECVAP19e', 
        'NHWHITEAMINCVAP19', 'NHWHITEAMINCVAP19e', 'NHWHITEASIANCVAP19', 'NHWHITEASIANCVAP19e', 
        'NHWHITEBLACKCVAP19', 'NHWHITEBLACKCVAP19e', 'NHBLACKAMINCVAP19', 'NHBLACKAMINCVAP19e', 
        'NHOTHCVAP19', 'NHOTHCVAP19e', 'HCVAP19e', 'POCCVAP19']

    Note that the ``acs5()`` method returns things like ``BCVAP19`` or ``HCVAP19`` (Black-alone
    CVAP and Hispanic CVAP, respectively) while the ``cvap()`` method returns things like
    ``NHBCVAP19`` (Non-Hispanic Black-alone CVAP). There are also columns like ``NHBCWVAP19``,
    which refer to all Non-Hispanic citizens of voting age who self-identified as Black
    and White. However, since our choice of :math:`Y` is restricted to single-race or ethnicity
    columns, we recommend only estimating CVAP for single-race or ethnicity
    columns, like ``BCVAP19``, ``HCVAP19``, or ``NHBCVAP19``).


    Lastly, one should choose :math:`Z` to match one's choice for :math:`Y` (say,
    ``BVAP20`` to match ``BVAP19``). However, in some cases it is reasonable to choose a :math:`Z`
    that is a close but imperfect match. For example, setting :math:`(X, Y, Z) =` 
    ``(BCVAP19, BVAP19, APBVAP20)`` (where :math:`Z =` ``APBVAP`` refers to all people of
    voting age who selected Black alone or in combination with other Census-defined races)
    would allow one to estimate the 2020 CVAP population of people who selected Black
    alone or in combination with other races.

    One final note: there are some instances in which, due to small Census reporting
    discrepancies, the ``acs5()`` and the ``cvap()`` methods disagree on ``CVAP19`` estimates
    (this might happen for total ``CVAP19`` or ``HCVAP19``, for example). In these cases
    we default to the ``acs5()`` numbers.

Now we may construct the estimated CVAP for 2020:

.. code-block:: python

    estimates = estimatecvap2010(
        base,
        us.states.AL,
        
        # Group order goes (Old CVAP, Old VAP, new VAP)
        groups=[
            ("WHITECVAP19", "WHITEVAP19", "WVAP20"),
            ("BLACKCVAP19", "BLACKVAP19", "BVAP20"),
        ],
        ceiling=1,
        zfill=0.1,
        geometry10="tract"
    )

The ``ceiling`` parameter marks when we will cap the CVAP / VAP ratio to 1. Set to 1,
this means that if there is ever more ``CVAP19`` in a geometry than ``VAP19`` , we
will "cap" the ``CVAP20`` estimate to 100\% of the ``VAP20`` . The ``zfill`` parameter
tells us what to do when there is 0 ``CVAP19`` in a geometry. Set to 0.1, this will
estimate that 10\% of the ``VAP20`` is ``CVAP``.

Now we can print our results:

.. code-block:: python

    print(f"Al BLACKCVAP20: {estimates.BLACKCVAP20_EST.sum()}")
    print(f"Al BLACKVAP19: {estimates.BLACKVAP19.sum()}")

Which returns to us:

.. code-block:: console

    AL BLACKCVAP20: 970120.3645540088
    AL BLACKCVAP19: 970239

We can see that our estimate for Black-alone Voting Age Population in Alabama in 2020
is 970,120, down slightly from 970,239 in 2019.

We can also make estimates of Black VAP in Alabama among ``APBVAP`` — Alabamians who
identified as Black alone or in combination with other races. This bumps up the
estimate to around 1,007,363 as we can see below:

.. code-block:: python 

    estimates = estimatecvap2010(
        base,
        us.states.AL,

        # Changing the new VAP column from BVAP20 -> APBVAP20
        groups=[
            ("BLACKCVAP19", "BLACKVAP19", "APBVAP20"),
        ],
        ceiling=1,
        zfill=0.1,
        geometry10="tract"
    )

    print(f"AL APBCVAP20 estimate: {estimates.BLACKCVAP20_EST.sum()}")

Which returns:

.. code-block:: console

    AL APBCVAP20 estimate: 1007362.5586538106