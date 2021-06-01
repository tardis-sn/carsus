***************
Units in Carsus
***************
 
In the recent years, much effort has been made to make Carsus easier to use and develop.
A large piece of code was rewritten to avoid using an intermediate SQL database between
the readers and the final atomic data. 

This resulted in an simpler, faster and more maintainable Carsus codebase. But we lost
some features in the process, being the most important one the tracking of physical
quantities. Currently, we couldn't solve the management of units in a sensible way.

We need to design and implement a new unit tracking system as soon as possible. In the
meantime, should be enough to document the units used by the GFALL/CHIANTI/CMFGEN readers
and the output module.
