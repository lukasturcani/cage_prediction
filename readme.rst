:author: Lukas Turcani

Introduction
============

This repo contains the code used for the paper "Machine Learning for
Organic Cage Property Prediction". In order to replicate the results
shown in the paper, go through the following steps:

1. Download the database of organic cages from
   https://doi.org/10.14469/hpc/4618. The cages are divided among
   multiple ``stk`` ``.json`` population dump files, one
   for each reaction. In addition, the SQL database used to
   get the results show in the paper is stored in
   ``cage_prediction.db``.
2. Extract the downloaded archive, ``cages.tar.gz``. For example,
   using ``tar -xzf cages.tar.gz``. This will extract the ``cages``
   folder, which holds the ``.json`` files and the SQL database.
3. Train the desired models. Please note that every training script has
   a usage statement which can be seen by running::

       python path_to_train_script.py --help

   for example::

       python train_scripts/collapse_prediction/random_forest.py --help

   Also note that in the commands below ``path/to/cage_prediction.db``
   should be replaced by the path to the file ``cage_prediction.db``
   on your computer.

   The script ``collapse_prediction/random_forest.py`` can be used to
   regenerate the results from Table 2 in the paper::

       python train_scripts/collapse_prediction/random_forest.py path/to/cage_prediction.db -r 1 2 3 4 5 6 -t 1

   The numbers after ``-r`` and ``-t`` indicate which reactions and
   topologies you wish to see the results for. By adding or removing
   numbers you can see the results for different rows of Table 2. To
   see which number corresponds to which reaction or topology run::

       python train_scripts/collapse_prediction/random_forest.py --help

   The script ``collapse_prediction/random_forest.py`` is also used to
   calculate the results for the cross-topology model with::

       python train_scripts/collapse_prediction/random_forest.py path/to/cage_prediction.db -r 1 2 8 9 10 11 12 -t 1 2 3 4 5 --join

   The script ``collapse_prediction/cross_reaction.py`` is used to
   get the results shown in Tables 3 and 4. To get the results for
   Table 3::

       python train_scripts/collapse_prediction/cross_reaction.py path/to/cage_prediction.db train 1 2 3 4 5 6

   and for Table 4::

       python train_scripts/collapse_prediction/cross_reaction.py path/to/cage_prediction.db test 1 2 3 4 5 6

Doing everything from scratch.
------------------------------

We provide the SQL database and optimized cage structures used in the
paper in https://doi.org/10.14469/hpc/4618. However, if you wish to
regenerate the results, starting only from the SMILES of the building
blocks and linkers, go through the following steps:

1. Generate the structures of the building blocks and linkers.
2. Assemble the unoptimized cages using ``stk``.
3. Optimize the structures of the cages, requires a MACROMODEL license.
4. Store the cage properties in a SQL database.


3. OPTIONAL: The SQL database can be remade by running
   ``./make_database.bash dirpath``, where ``dirpath`` is the path
   to the ``cages`` folder extracted from ``cages.tar.gz``. Before
   the database can be recalculated, you have to install ``stk`` and
   ``rdkit``. Installation of these two libraries is only necessary if
   you wish to regenerate the SQL database, it is not necessary to
   train any of the machine learning models. To install these libraries
   see: `Installing rdkit and stk`_.


Installing rdkit and stk.
-------------------------

Note that this is only necessary if you are recalculating the cage
properties. A SQL database holding the calculated cage properties
can be downloaded from https://doi.org/10.14469/hpc/4618 and is
held in the file ``cage_prediction.db``.

Make sure you are using the Anaconda distribution of Python. This
is necessary because ``stk`` depends on ``rdkit``, which requires the
conda package manager. ``rdkit`` can be installed without the
Anaconda distribution but it is a significantly more complicated
process. If you wish to do it anyway, refer to the ``rdkit``
documentation, https://github.com/rdkit/rdkit, for help. If have
Anaconda Python installed, just type the following commands into your
terminal:

1. ``conda install -c rdkit rdkit``
2. ``pip install stk``

Files
=====

The files used for this are held in the following folders: ``database``,
``train_scripts``, ``trained_models`` and ``website``. The
``database`` folder contains code which is used to create the SQL
database holding the properties of organic cages used in this study.
The ``train_scripts``
folder contains scripts which use the SQL database to train random
forest models for cage property prediction. ``trained_models`` contains
pickled scikit-learn random forest estimators which have been trained.
These are the models which the website, https://ismycageporous.ngrok.io, uses.
The ``website`` folder contains the code to make the aforementioned website.

:database/make_database.bash:
:database/make_database.py:
:
