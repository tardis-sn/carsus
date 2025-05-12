***************
Differences between level_index, level_id and level_number
***************

=============
level_index
=============

DataFrame index for each electronic energy level. Different for each ion. 
Third index for energy levels after atomic number and ion charge. Used in GFALL,
Chianti and CMFGEN levels DataFrames.

For photoionization cross-sections, this is used to determine the lower and upper
levels of cross-sections.

=============
level_id
=============

Global unique `level_id` across all species used to assign unique IDs even for the cut level and line data.
Computed from level index. Typically has an upper and lower form to connect levels
together.

=============
level_number
=============

Index in the order of the level energies within each species. 
For example, species Si II has level_number 0 to n. 
Often used in a multi-index with atomic_number, ion_number.
Connects upper and lower energy levels.