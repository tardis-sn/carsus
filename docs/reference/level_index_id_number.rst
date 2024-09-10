***************
Differences between level_index, level_id and level_number
***************

=============
level_index
=============

DataFrame index for each electronic energy level. Different for every level. 
Third index for energy levels after atomic number and ion charge. 

=============
level_id
=============

Global cross-matching ID for levels across multiple DataFrames i.e. between photo
ionization cross sections, the macro atom and electronic energy levels.

=============
level_number
=============

DataFrame index for each level suitable for TARDIS use. Connects upper and lower
energy levels.