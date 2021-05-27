*********************************
Notation in Carsus
*********************************

* **"0" for neutral elements.** 
    ``Si 0``  is equivalent to :math:`\text{Si I}`, ``Si 1`` to :math:`\text{Si II}`, etc.

* **"-" to grab intervals of consectutive elements or species.**
    ``H-He`` selects  :math:`\text{H I}` and :math:`\text{H II}` plus :math:`\text{He I}`,  :math:`\text{He II}` and  :math:`\text{He III}`, while ``C 0-2`` selects  :math:`\text{C I}`,  :math:`\text{C II}` and :math:`\text{C III}`. 

* **"," to grab non-consecutive species.** 
    ``Si 0, 2`` selects :math:`\text{Si I}` and :math:`\text{Si III}`.
  
* **";" to grab non-consecutive elements.**
    ``H; Li`` selects  :math:`\text{H I}` and :math:`\text{H II}` plus :math:`\text{Li I}`,  :math:`\text{Li II}`, :math:`\text{Li III}` and :math:`\text{Li IV}`.

* **Mix all the above syntax as needed.**
    ``H; C-Si; Fe 1,3``.