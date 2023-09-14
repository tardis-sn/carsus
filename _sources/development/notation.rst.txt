******************
Notation in Carsus
******************

* **Use "0" for neutral elements.** 
    Example: ``Si 0``  is equivalent to :math:`\text{Si I}`, ``Si 1`` to :math:`\text{Si II}`, etc.

* **Use "-" to grab intervals of consecsutive elements or species.**
    Example: ``H-He`` selects  :math:`\text{H I}` and :math:`\text{H II}` plus :math:`\text{He I}`,  :math:`\text{He II}` and  :math:`\text{He III}`, while ``C 0-2`` selects  :math:`\text{C I}`,  :math:`\text{C II}` and :math:`\text{C III}`. 

* **Use "," to grab non-consecutive species.** 
    Example: ``Si 0, 2`` selects :math:`\text{Si I}` and :math:`\text{Si III}`.
  
* **Use ";" to grab non-consecutive elements.**
    Example: ``H; Li`` selects  :math:`\text{H I}` and :math:`\text{H II}` plus :math:`\text{Li I}`,  :math:`\text{Li II}`, :math:`\text{Li III}` and :math:`\text{Li IV}`.

* **Finally, mix all the above syntax as needed.**
    Example: ``H; C-Si; Fe 1,3``.