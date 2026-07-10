Atomic Data Reproduction Differences
====================================

This page records the observed differences between the TARDIS regression
reference atom-data file and the Carsus-generated reproduction candidate.

Files compared
--------------

Reference file:
  ``/home/afullard/tardis-regression-data/atom_data/kurucz_cd23_chianti_H_He_latest.h5``

Generated file:
  ``/tmp/carsus_reference_experiment.h5``

The generated file was built from the recovered historical Kurucz flat
``gfall.dat`` in ``tardis-sn/carsus-data-kurucz``, the NNDC CSV snapshot from
``tardis-sn/carsus-data-nndc`` commit
``136a8633e3dee21079d3738dabfe7758f6e40d41``, and CHIANTI H-He data with
CHIANTI 7.1 ``.wgfa`` radiative transition rows overlaid onto the current
CHIANTI reader output.

The flat file regenerates the classic SQLite ``gfall`` table content. In a
controlled Carsus build where only the GFALL reader was changed, the flat file
and the classic SQLite table produced exactly equal levels, lines, macro atom
data, and macro atom references. SQLite is therefore not part of the
reproduction path.

High-level result
-----------------

Both files contain the same HDF keys. No table has reference-only or
generated-only index rows. The following tables compare exactly with
``DataFrame.equals``:

* ``/atom_data``
* ``/collisions_metadata``
* ``/ionization_data``
* ``/levels_data``
* ``/macro_atom_references``
* ``/zeta_data``

The following tables do not compare byte-for-byte:

* ``/collisions_data``
* ``/decay_radiation_data``
* ``/lines_data``
* ``/macro_atom_data``
* ``/metadata``

HDF keys and shapes
-------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Reference shape
     - Generated shape
     - Exact
     - Index equal
     - Reference-only rows
     - Generated-only rows
   * - ``/atom_data``
     - ``(30, 3)``
     - ``(30, 3)``
     - yes
     - yes
     - 0
     - 0
   * - ``/collisions_data``
     - ``(358, 14)``
     - ``(358, 14)``
     - no
     - yes
     - 0
     - 0
   * - ``/collisions_metadata``
     - ``(2,)``
     - ``(2,)``
     - yes
     - yes
     - 0
     - 0
   * - ``/decay_radiation_data``
     - ``(242295, 41)``
     - ``(242295, 41)``
     - no
     - yes
     - 0
     - 0
   * - ``/ionization_data``
     - ``(465,)``
     - ``(465,)``
     - yes
     - yes
     - 0
     - 0
   * - ``/levels_data``
     - ``(24806, 3)``
     - ``(24806, 3)``
     - yes
     - yes
     - 0
     - 0
   * - ``/lines_data``
     - ``(271741, 8)``
     - ``(271741, 8)``
     - no
     - no, in stored order
     - 0
     - 0
   * - ``/macro_atom_data``
     - ``(815223, 7)``
     - ``(815223, 7)``
     - no
     - yes
     - 0
     - 0
   * - ``/macro_atom_references``
     - ``(24806, 3)``
     - ``(24806, 3)``
     - yes
     - yes
     - 0
     - 0
   * - ``/metadata``
     - ``(18, 1)``
     - ``(18, 1)``
     - no
     - yes
     - 0
     - 0
   * - ``/zeta_data``
     - ``(412, 20)``
     - ``(412, 20)``
     - yes
     - yes
     - 0
     - 0

Root attributes
---------------

``FORMAT_VERSION`` and ``database_version`` match. The timestamp, UUID, and
file MD5 differ because the generated file was written in a different run.

.. list-table::
   :header-rows: 1

   * - Attribute
     - Reference
     - Generated
     - Equal
   * - ``FORMAT_VERSION``
     - ``2.0``
     - ``2.0``
     - yes
   * - ``database_version``
     - ``v0.9``
     - ``v0.9``
     - yes
   * - ``DATE``
     - ``2025-05-28T19:19:43.222133+00:00``
     - ``2026-06-30T14:07:39.073939+00:00``
     - no
   * - ``UUID1``
     - ``b58b2ef63bf811f08edf96479f911fbd``
     - ``0d7d7132748d11f183b469a0031257ca``
     - no
   * - ``MD5``
     - ``5d80fa4ae0638469bf1ff281b6ca2a94``
     - ``e9f4e1d62283c8d72460f830f6315f57``
     - no

``/lines_data``
---------------

After sorting both tables by their physical index
``(atomic_number, ion_number, level_number_lower, level_number_upper)``, all
rows align. The remaining differences are concentrated in H I, He I, and He II
line IDs, plus floating-point roundoff in quantities derived from the CHIANTI
line values.

``line_id`` differs for 321 rows. The maximum absolute line-ID shift is 3.

.. list-table::
   :header-rows: 1

   * - Species index
     - Differing rows
   * - ``(1, 0)``
     - 74
   * - ``(2, 0)``
     - 180
   * - ``(2, 1)``
     - 67

Column differences after index alignment, expressed relative to the reference
value:

.. list-table::
   :header-rows: 1

   * - Column
     - Differing rows
     - Max relative difference
     - Mean relative difference
   * - ``line_id``
     - 321
     - ``5.624834770478617e-06``
     - ``4.381902051173341e-09``
   * - ``wavelength``
     - 321
     - ``3.5296725691438195e-16``
     - ``2.258608679965486e-19``
   * - ``f_ul``
     - 92
     - ``2.867333931901698e-16``
     - ``6.130846763437799e-20``
   * - ``f_lu``
     - 86
     - ``2.867333931901698e-16``
     - ``5.808156081243093e-20``
   * - ``nu``
     - 270
     - ``4.5169652833628e-16``
     - ``2.1890800632067765e-19``
   * - ``B_lu``
     - 327
     - ``5.666398681217063e-16``
     - ``2.6988645838974265e-19``
   * - ``B_ul``
     - 335
     - ``5.666398681217063e-16``
     - ``2.715503347390916e-19``
   * - ``A_ul``
     - 543
     - ``1.1282367931533941e-15``
     - ``6.58566534257689e-19``

The line values are therefore physically aligned to numerical precision. For
example, the largest aligned ``B_ul`` relative difference is
``5.666398681217063e-16``. The remaining line-ID shifts are a legacy numbering
artifact from the order and count of rows present before source-priority
filtering.

``/collisions_data``
--------------------

The collision table has the same shape and physical index in both files. All
358 rows differ in the internal ID columns. The collision energies differ at
small numerical levels.

.. list-table::
   :header-rows: 1

   * - Column
     - Differing rows
     - Max relative difference
     - Mean relative difference
   * - ``e_col_id``
     - 358
     - ``4.686527545534302e-05``
     - ``4.6849600490378205e-05``
   * - ``lower_level_id``
     - 358
     - ``0.011138521342602976``
     - ``0.011126068467180832``
   * - ``upper_level_id``
     - 358
     - ``0.01113810502709774``
     - ``0.01111742439020622``
   * - ``energy_lower``
     - 262
     - ``4.446661999562916e-08``
     - ``4.4466619847822936e-08``
   * - ``energy_upper``
     - 358
     - ``4.446662010541005e-08``
     - ``4.4466619856192806e-08``
   * - ``delta_e``
     - 358
     - ``5.732640138846471e-08``
     - ``5.732638835087047e-08``

``/macro_atom_data``
--------------------

The macro atom table has the same shape and index in both files. Comparing in
stored index order gives large apparent differences because row ordering within
source-level groups depends on line IDs. After sorting by all physical macro
atom columns, the residual differences are limited to H-He line-ID shifts and
floating-point roundoff.

Stored-index comparison, expressed relative to the reference value. This is a
row-order diagnostic rather than a physical comparison because rows within
source-level groups are not aligned:

.. list-table::
   :header-rows: 1

   * - Column
     - Differing rows
     - Max relative difference
     - Mean relative difference
   * - ``destination_level_number``
     - 563325
     - 837.0
     - ``1.1245320224132056``
   * - ``transition_type``
     - 252371
     - 2
     - ``0.47004868606503986``
   * - ``transition_probability``
     - 574050
     - ``370354065608758.7``
     - ``1705204017.1841183``
   * - ``transition_line_id``
     - 564141
     - ``0.6732522796352584``
     - ``0.008082544296154417``

Content-sorted comparison, expressed relative to the reference value:

.. list-table::
   :header-rows: 1

   * - Column
     - Differing rows
     - Max relative difference
     - Mean relative difference
   * - ``transition_probability``
     - 1290
     - ``1.0448848738309492e-15``
     - ``4.887465866414813e-19``
   * - ``transition_line_id``
     - 963
     - ``5.624834770478617e-06``
     - ``4.381902051173341e-09``

``/decay_radiation_data``
-------------------------

The decay radiation table has the same shape, columns, and isotope index in
both files. Differences are serialization and floating-point representation
differences from parsing the same NNDC CSV snapshot with newer dependencies.

Numeric column differences:

.. list-table::
   :header-rows: 1

   * - Column
     - Differing rows
     - Max relative difference
     - Mean relative difference
   * - ``Decay Mode Value``
     - 1295
     - ``2.085004177856739e-16``
     - ``7.578287110737619e-19``
   * - ``Q Value``
     - 19
     - ``1.6802467266366348e-16``
     - ``9.685144611315847e-21``
   * - ``T1/2 (sec)``
     - 7535
     - ``5.927827262583468e-14``
     - ``1.1967362187028166e-17``
   * - ``Rad Energy``
     - 7984
     - ``2.2197047025625434e-16``
     - ``5.014792926927897e-18``
   * - `` EP Energy``
     - 4
     - ``1.9109857881773446e-16``
     - ``3.0332460912299135e-20``
   * - ``Rad Intensity``
     - 31067
     - ``2.207027323125709e-16``
     - ``1.8993832595513687e-17``
   * - ``Dose``
     - 44861
     - ``3.0691244944768855e-16``
     - ``2.758227980134172e-17``

Object/string-like column differences:

.. list-table::
   :header-rows: 1

   * - Column
     - Differing rows
   * - ``Parent E(level)``
     - 672
   * - ``Gammas Balance``
     - 2311
   * - ``B- Balance``
     - 621
   * - ``B+ Balance``
     - 624
   * - ``Conversion Electrons Balance``
     - 2040
   * - ``Auger Electrons Balance``
     - 24
   * - ``Recoil Balance``
     - 79
   * - `` Protons Balance``
     - 1
   * - ``Alphas Balance``
     - 73
   * - ``Sum Balance``
     - 111
   * - ``Q-effective Balance``
     - 9
   * - ``Missing Energy Balance``
     - 20

``/metadata``
-------------

The metadata table shape and index match, but 17 of 18 rows differ. The
differences are expected because the file was generated with a newer Carsus
checkout and newer runtime dependencies, and because each table's stored MD5
is recomputed from the generated representation.

.. list-table::
   :header-rows: 1

   * - Field
     - Key
     - Reference
     - Generated
   * - ``md5sum``
     - ``atom_data``
     - ``7214216cc5cfda76089f261c37212a04``
     - ``1c075be4454adb32eb59ccf96e540cd5``
   * - ``md5sum``
     - ``collisions_data``
     - ``c2e2ad0009fd3388d97b72e4a3cf8601``
     - ``68e262dcc4be0ca5cc18ed06ce601645``
   * - ``md5sum``
     - ``collisions_metadata``
     - ``1a845db11b82c8da75a22cfb5d0f0734``
     - ``33a56c0abaec5ec37473c130e9e68ff6``
   * - ``md5sum``
     - ``decay_radiation_data``
     - ``e3ae85386dd5d4a86ebf09bc8563c7de``
     - ``567484a1aff18ccb8b8afce542a14f6c``
   * - ``md5sum``
     - ``ionization_data``
     - ``8eaf0cb822408b97ae018eaf5b8c1d4b``
     - ``7bfc5466b0cd9896f4ca1f386dc73a19``
   * - ``md5sum``
     - ``levels``
     - ``eb7f50552d79e0fcda6daa289de669a7``
     - ``f0117ed27b7a2eebc4898a986613d477``
   * - ``md5sum``
     - ``lines``
     - ``b0a3d64dd607522886a0c9e722254470``
     - ``3bba6940617d76a343b4ed6c9a951f07``
   * - ``md5sum``
     - ``macro_atom_data``
     - ``745e7d7b23f52af2c170bec0f03ccf25``
     - ``34817c05690a02e7142d8e038e666781``
   * - ``md5sum``
     - ``macro_atom_references``
     - ``86dc1a2668625a1bf4c6c912f32f02a1``
     - ``e542f4f82013cbf611927dc26177a9dc``
   * - ``md5sum``
     - ``zeta_data``
     - ``2c9ac05732b3678212075b77cdf2ba62``
     - ``3555b5c9000ae58eb92ac5d0018ca360``
   * - ``software``
     - ``ChiantiPy``
     - ``0.15.0``
     - ``0.16.0``
   * - ``software``
     - ``astropy``
     - ``6.1.3``
     - ``7.2.0``
   * - ``software``
     - ``carsus``
     - ``2024.12.24.dev43+g089048e.d20250528``
     - ``2024.12.24.dev64+gaf7c224cb.d20260630``
   * - ``software``
     - ``numpy``
     - ``2.1.1``
     - ``2.4.6``
   * - ``software``
     - ``pandas``
     - ``2.2.2``
     - ``3.0.3``
   * - ``software``
     - ``python``
     - ``3.12.5``
     - ``3.14.5``
   * - ``software``
     - ``tables``
     - ``3.10.1``
     - ``3.11.1``

Interpretation
--------------

The recovered source data are sufficient to reproduce the reference file's
schema, table set, row counts, indexes, and all core level/ion/zeta/atom tables.
The remaining differences are not missing source rows. They fall into four
categories:

* run-specific root attributes and table MD5 signatures;
* dependency/version metadata;
* floating-point and object serialization differences from newer pandas,
  PyTables, NumPy, Astropy, and ChiantiPy versions;
* legacy line-ID numbering differences in the CHIANTI H-He line block, which
  propagate into ``/macro_atom_data`` because macro atom rows store
  ``transition_line_id``.
