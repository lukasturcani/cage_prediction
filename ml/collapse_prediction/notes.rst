16/1/2018
=========

Project status so far.

I've modified molder so that I can tag cages as being collapsed or not.

The modifications included storing both the cage and the building blocks
which compose it in the database.json and changing the question. An
additional button was also added which tages cages and "unsure". This
is so that my training set is composed of very clearly collapsed or
not collapsed cages.

For the cages, I'm using liverpool_refined, using the databases
optimized on  06/2017. The database for molder was created using
documents/molder/create_cage_collapse_database.py

Some cages were automatically tagged by using
documents/molder/collapse_autotag

The remaining ones I'm going through as we speak.

18/1/2018
=========

Finished tagging rest of liverpool_refined.

19/1/2018
=========

Going to make a prediction using SKlearn on the tagged liverpool_refined
data.

Going to use the script ``make_h5.py`` to read the json database and
convert cages to fingerprints and extract the labels and dump them
to a .h5 file.


Going to fit an svm using sklearn, ``fit_svm.py``.
It worked! getting 0.87 with cross validation fold of 10.

25/1/2018
=========

I've saved the previous collection setup into og_collection_setup and
this has been placed into Dropbox.

I will now repeat exactly what I did before but using the reoptimized
liverpool_refined structures.

12/2/2018
=========

Done relabelling. Getting 0.87 using 10 fold and 0.84 using 5 fold CV.
Made various input features using different fingerprint sizes and
fingerprint radii.

13/2/2018
=========

Created the scripts ``create_featurizations.bash``,
``analysis/calc_featurization_heatmap.py``,
``analysis/featurization_heatmap.py``
and ``analysis/bb_ranking.py``.

``create_featurizations.bash`` creates various input featurizations for
the prediciton task. It creates featurizations using cage fingerprints,
joined building block fingerprints and joined cage and building block
fingerprints. It also varies the fingerprint radius and bit size.

``analysis/calc_featurization_heatmap.py`` fits an SVM using each of
the featurizations and dumps the results using a pandas DataFrame.

``analysis/featurization_heatmap.py`` plots those results.

``analysis/bb_ranking.py`` goes through the building blocks and counts
how many times each building block appears in a collapsed and not
callapsed cage. It then writes these building blocks into approriate files.

14/2/2018
=========

Created ``analysis/grid_bbs.pml`` which can be used from any of the
folders created by ``analysis/bb_ranking.py``. This will create a pymol
image showing the building blocks in that folder on a grid. This
should be run from the given folder.

19/4/2018
=========

I want to completely automate tagging of molecules. The approach will be
to create a function which can identify collapsed molecules and one
which can identify non - collapsed one. Some cages will not fall into
either category. The point is to just get the clearly collapsed and
clearly not collapsed molecules.

Edited ``sort_databases.py`` to carry out the categorization as described
in the paragraph above.

20/4/2018
=========

Looks like pywindow works really well for finding collapsed cages.
Only 9 examples were classified as collapsed when they were not really.
Playing with the EPS parameter can be the solution, but this might not
work for cages of different topologies.

Next thing to check if overlap between pywindow collapsed and ones I
said were collapsed and / or undetermined. Then I can do classification
again.


27/4/2018
=========

I'm starting almost from scratch. I deleted or moved all the old training
scripts and the scripts used to make featurizations. Everything has been
moved to MongoDB so all the new scripts will take this into account
and use data from there.


4/6/2018
========

I've created a new database called "small". It contains the cores in
liverpool_refined but all cores are in all functionalities. Small
also consists of more functional groups. I created a bunch of cages
with different reactions and different topologies using
"documents/stk/molecule_generation/small.py". I optimized them and
added them to the MongoDB, database = small, collection = cages. The
building blocks were placed in database = small, collection = bbs.
Each cage has a tag indicating which reaction was used to create it.


I calculated their "cage+bb" fingerprints and collapse categorization using
"pywindow_plus". The cage fingerprint was calculed on the unoptimzed
cage structures. This was done using "update_database.py" and the
input files "add_fingerprints.py" and "add_collapse_labels.py"

I will now examine the performance of various models on collapse
prediction. To do examination of this - each cage fingerprint must be
extended with a 1 hot representation of its topology.

ALL THESE RESULTS ARE WRONG BECAUSE I USED ONLY BB FINGERPRINTS AND
LABELLED IT AS CAGE+BB

vanilla
    Group all generated cages into 1 dataset and run cross validation.

    The results were:

        0.848412902416

        [[2499  377]
        [1120 5885]]

                 precision    recall  f1-score   support

              0       0.69      0.87      0.77      2876
              1       0.94      0.84      0.89      7005

        avg / total       0.87      0.85      0.85      9881


    Next I can do this reaction by reaction.

    amine2aldehyde3



        FourPlusSix only.
        -----------------

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({0: 2314, 1: 2269})
        0.864720667894

        [[402  61]
         [ 45 409]]

                     precision    recall  f1-score   support

                  0       0.90      0.87      0.88       463
                  1       0.87      0.90      0.89       454

        avg / total       0.88      0.88      0.88       917


        EightPlusTwelve only.
        ---------------------

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 4249, 0: 1223})
        0.839721960913

        [[215  30]
         [153 697]]

                     precision    recall  f1-score   support

                  0       0.58      0.88      0.70       245
                  1       0.96      0.82      0.88       850

        avg / total       0.87      0.83      0.84      1095


        Both.
        -----

        Counter({1: 6518, 0: 3537})
        0.85350638268

        [[ 601  106]
         [ 195 1109]]

                     precision    recall  f1-score   support

                  0       0.76      0.85      0.80       707
                  1       0.91      0.85      0.88      1304

        avg / total       0.86      0.85      0.85      2011



    aldehyde2amine3

        FourPlusSix only.
        -----------------


        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2


        Counter({1: 2445, 0: 2206})
        0.862178256702

        [[398  44]
         [ 82 407]]

                     precision    recall  f1-score   support

                  0       0.83      0.90      0.86       442
                  1       0.90      0.83      0.87       489

        avg / total       0.87      0.86      0.86       931


        EightPlusTwelve only.
        ---------------------

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 4395, 0: 1102})
        0.865558276119

        [[190  31]
         [117 762]]

                     precision    recall  f1-score   support

                  0       0.62      0.86      0.72       221
                  1       0.96      0.87      0.91       879

        avg / total       0.89      0.87      0.87      1100

        Both.
        -----

        Counter({1: 6840, 0: 3308})
        0.873276408335

        [[ 585   77]
         [ 187 1181]]

                     precision    recall  f1-score   support

                  0       0.76      0.88      0.82       662
                  1       0.94      0.86      0.90      1368

        avg / total       0.88      0.87      0.87      2030




    aldehyde4amine2

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 1510, 0: 780})
        0.784716157205

        [[138  18]
         [ 76 226]]

                     precision    recall  f1-score   support

                  0       0.64      0.88      0.75       156
                  1       0.93      0.75      0.83       302

        avg / total       0.83      0.79      0.80       458


    aldehyde4amine3

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 549, 0: 471})
        0.824517719619

        [[86  8]
         [27 83]]

                     precision    recall  f1-score   support

                  0       0.76      0.91      0.83        94
                  1       0.91      0.75      0.83       110

        avg / total       0.84      0.83      0.83       204


    alkene2alkene3

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 2529, 0: 2026})
        0.843472010686

        [[334  71]
         [ 64 442]]

                     precision    recall  f1-score   support

                  0       0.84      0.82      0.83       405
                  1       0.86      0.87      0.87       506

        avg / total       0.85      0.85      0.85       911


    amine2carboxylic_acid3

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 3972, 0: 1105})
        0.76954966836

        [[171  50]
         [177 618]]

                     precision    recall  f1-score   support

                  0       0.49      0.77      0.60       221
                  1       0.93      0.78      0.84       795

        avg / total       0.83      0.78      0.79      1016


    amine3aldehyde3

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 1250, 0: 858})
        0.718217739303

        [[132  40]
         [ 67 183]]

                     precision    recall  f1-score   support

                  0       0.66      0.77      0.71       172
                  1       0.82      0.73      0.77       250

        avg / total       0.76      0.75      0.75       422


    thiol2thiol3

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 5700, 0: 99})
        0.827216387492

        [[  8  12]
         [176 964]]

                     precision    recall  f1-score   support

                  0       0.04      0.40      0.08        20
                  1       0.99      0.85      0.91      1140

        avg / total       0.97      0.84      0.90      1160


    alkyne22alkyne23

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({0: 3148, 1: 1981})
        0.912653221758

        [[566  64]
         [ 29 367]]

                     precision    recall  f1-score   support

                  0       0.95      0.90      0.92       630
                  1       0.85      0.93      0.89       396

        avg / total       0.91      0.91      0.91      1026


    amine4aldehyde2

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 1525, 0: 769})
        0.786836772555

        [[137  17]
         [ 77 228]]

                     precision    recall  f1-score   support

                  0       0.64      0.89      0.74       154
                  1       0.93      0.75      0.83       305

        avg / total       0.83      0.80      0.80       459


    amine4aldehyde3

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2


        Counter({1: 519, 0: 500})
        0.806664734859

        [[84 16]
         [21 83]]

                     precision    recall  f1-score   support

                  0       0.80      0.84      0.82       100
                  1       0.84      0.80      0.82       104

        avg / total       0.82      0.82      0.82       204


    carboxylic_acid2amine3

        .. code-block:: python

            def pywindow_plus(match, calc_params):
                struct = next(s for s in match['structures'] if
                              s['calc_params'] == calc_params)
                c = Cage(match, calc_params)

                wd = struct['window_difference']
                md = struct['max_diameter']
                cs = struct['cavity_size']

                if wd is None:
                    return 1
                elif (4*wd)/(md*c.topology.n_windows) < 0.035 and cs > 1:
                    return 0
                else:
                    return 2

        Counter({1: 3724, 0: 1368})
        0.815198967396

        [[230  44]
         [158 587]]

                     precision    recall  f1-score   support

                  0       0.59      0.84      0.69       274
                  1       0.93      0.79      0.85       745

        avg / total       0.84      0.80      0.81      1019


topology
    Separate building block pairs into training and test pairs. For each
    training pair train on all cages of all topologies made from that pair
    for each test pair test on all cages of all topologies made from that
    pair.

reverse_reaction
    For example, train on amine2aldehyde3 and predict on aldehyde3amine2.

cross_reaction
    For example, train on amine2aldehyde3 and prediction on thiol2thiol3.


6/6/2018
========

Still filling in that table with results. However I need to change the
labeller as I dont think the settings which work for 4+6 also work
for the other topologies. As a result, I am precalculating all the
values of cages and adding them into the database using
"add_struct_params.py".

The script "label_check.py" allows me to quickly write down the structure
files of collapsed and non collapsed cages.

Some reactions lead to highly imbalanced datasets - for example thiol2thil3
produces 5676 collapsed and 93 non collapsed. Looking at the apparently
non collapsed ones - they are all actually collapsed.


18/6/2018
=========

Do the predictions above but with RandomForest and 1 class SVM.
Results saved into ...
