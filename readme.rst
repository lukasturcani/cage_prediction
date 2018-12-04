:author: Lukas Turcani

    Please note that results in the paper were calculated using the
    Anaconda distribution of Python 3.6.3 downloaded from
    https://repo.anaconda.com/archive/Anaconda3-5.0.1-Linux-x86_64.sh.
    I noticed that even switching to a different version of Python 3.6,
    such as 3.6.6, introduced some very slight, and completely negligible,
    differences in some of the numbers reported in the paper.
    While this should be of no consequence, it is pretty annoying

Introduction
============

This repo contains the code used for the paper "Machine Learning for
Organic Cage Property Prediction". In order to replicate the results
shown in the paper, go through the following steps:

1. Download the file ``cage_prediction.db`` from
   https://doi.org/10.14469/hpc/4618. This is a SQL database holding
   the cage properties which are modelled in the paper.
2. Train the desired models using the files in ``train_scripts``.
   Note that every training script has
   a usage statement which can be seen by running::

       python path_to_train_script.py --help

   for example::

       python train_scripts/collapse_prediction/random_forest.py --help

   Also note that in the commands below ``path/to/cage_prediction.db``
   should be replaced by the path to the file ``cage_prediction.db``
   on your computer.

``collapse_prediction/random_forest.py``
----------------------------------------

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

``collapse_prediction/cross_reaction.py``
-----------------------------------------

The script ``collapse_prediction/cross_reaction.py`` is used to
get the results shown in Tables 3 and 4. To get the results for
Table 3::

   python train_scripts/collapse_prediction/cross_reaction.py path/to/cage_prediction.db train 1 2 3 4 5 6

and for Table 4::

   python train_scripts/collapse_prediction/cross_reaction.py path/to/cage_prediction.db test 1 2 3 4 5 6

``regression/results_table.py``
-------------------------------

The script ``regression/results_table.py`` can be used to make
Table 5 and 6 and Table 2 in the SI. For example::

   python train_scripts/regression/results_table.py path/to/cage_prediction.db cage_property

where ``cage_property`` can be either ``window_diff``, ``window_std``
or ``cavity_size``. Note that this script prints the
results in a Latex syntax.

``regression/random_forest.py`` and ``regression/cross_reaction.py``
--------------------------------------------------------------------

The scripts ``regression/random_forest.py``
and ``regression/cross_reaction.py`` can be used to get the results for
individual rows of Tables 5, 6 and Table 2 in the SI::

   python train_scripts/regression/random_forest.py path/to/cage_prediction.db cage_property -r 1 2 3 -t 1
   python train_scripts/regression/random_forest.py path/to/cage_prediction.db cage_property 1 2 3 4 5

Note that these scripts are run exactly like the ``collapse_prediction/random_forest.py`` and
``collapse_prediction/cross_reaction.py`` with the exception that
``window_diff``, ``window_std`` or ``cavity_size`` must be specified
after ``path/to/cage_prediction.db``. For example, to get the
results of the cross-topology model for cavity sizes::

   python train_scripts/regression/random_forest.py path/to/cage_prediction.db cavity_size -r 1 2 8 9 10 11 12 -t 1 2 3 4 5 --join


Doing everything from scratch.
==============================

We provide the SQL database and optimized cage structures used in the
paper in https://doi.org/10.14469/hpc/4618. However, if you wish to
regenerate the results, starting only from the SMILES of the building
blocks and linkers, go through the following steps:

Installing rdkit and stk.
-------------------------

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

Generating the SQL database.
----------------------------

To generate the database from the SMILES strings go through the
follow steps 1 to 4. If you want to skip remaking the cage molecules
and re-optimizing them, you can download the ``.json`` holding the
optimized cages from https://doi.org/10.14469/hpc/4618 and go straight
to step 4. This will use the optimized cages and recalculate their
properties.


1. Generate the structures of the building blocks and linkers::

       python create_structs.py

2. Assemble the unoptimized cages using ``stk``::

       python assemble.py 1 2 3 5 6 7 8 11 18 19 26 27

3. Optimize the structures of the cages, requires a MACROMODEL license.
   The repository https://github.com/lukasturcani/chem_tools
   has a script called ``optimize.py``, which can easily optimize
   molecules in a ``stk`` population file. This can make the optimization
   step significantly easier. Note that this step can take multiple
   days. For example,  to optimize the structures of the cages with
   in the ``amine2aldehyde3.json`` file with ``optimize.py``::

       python optimize.py amine2amine2aldehyde3.json settings.py amine2aldehyde3_opt.json /opt/schrodinger2017-4

   Run::

       python optimize.py --help

   for an explanation of the command line arguments. It may also help
   to read the docstring within the file.

4. Store the cage properties in a SQL database. The SQL database can be
   remade by running::
       ./make_database.bash dirpath

   where ``dirpath`` is the path
   to the ``cages`` folder extracted from ``cages.tar.gz``, which is
   downloaded from https://doi.org/10.14469/hpc/4618.
   ``make_database.bash`` if found in the ``database`` folder of this
   repository.

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
