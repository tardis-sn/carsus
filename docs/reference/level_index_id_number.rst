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

Global cross-matching ID for levels across multiple DataFrames i.e. between photo
ionization cross sections, the macro atom and electronic energy levels. 
Computed from level index. Typically has an upper and lower form to connect levels
together.

=============
level_number
=============

DataFrame index for each electronic energy level suitable for TARDIS use. 
Connects upper and lower energy levels.