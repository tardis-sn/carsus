import logging
import pathlib

import astropy.units as u
import numpy as np
import pandas as pd
import roman
import yaml
from scipy import interpolate

from carsus import __path__ as CARSUS_PATH
from carsus.io.base import BaseParser
from carsus.util import convert_atomic_number2symbol, parse_selected_species

from .util import *

logger = logging.getLogger(__name__)


class CMFGENEnergyLevelsParser(BaseParser):
    """
    Description
    ----------
    base : pandas.DataFrame
    header : dict

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    def load(self, fname):
        header = parse_header(fname)
        skiprows, _ = find_row(fname, "Number of transitions")
        nrows = int(header["Number of energy levels"])
        config = {
            "header": None,
            "index_col": False,
            "sep": "\s+",
            "skiprows": skiprows,
            "nrows": nrows,
            "engine": "python",
        }

        try:
            df = pd.read_csv(fname, **config)

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            logger.warning(f"Empty table: `{fname}`.")

        if df.shape[1] == 10:
            # Read column names and split them keeping just one space (e.g. '10^15 Hz')
            _, columns = find_row(fname, "E(cm^-1)", "Lam")
            columns = columns.split("  ")
            columns = [c.rstrip().lstrip() for c in columns if c != ""]
            columns = ["label"] + columns
            df.columns = columns

        elif df.shape[1] == 7:
            df.columns = ["label", "g", "E(cm^-1)", "eV", "Hz 10^15", "Lam(A)", "ID"]

        elif df.shape[1] == 6:
            df.columns = ["label", "g", "E(cm^-1)", "Hz 10^15", "Lam(A)", "ID"]

        elif df.shape[1] == 5:
            df.columns = ["label", "g", "E(cm^-1)", "eV", "ID"]

        else:
            logger.warning(f"Unknown column format: `{fname}`.")

        self.base = df
        self.header = header
        # Re-calculate Lam(A) values
        self.base["Lam(A)"] = self.calc_Lam_A()

    def calc_Lam_A(self):
        """
        Calculate and replace column `Lam(A)`.
        """

        level_ionization_threshold = (
            float(self.header["Ionization energy"]) - self.base["E(cm^-1)"]
        )

        return (level_ionization_threshold.values / u.cm).to(
            "Angstrom", equivalencies=u.spectral()
        )


class CMFGENOscillatorStrengthsParser(BaseParser):
    """
    Description
    ----------
    base : pandas.DataFrame
    header : dict

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    def load(self, fname):
        header = parse_header(fname)
        skiprows, _ = find_row(fname, "Transition", "Lam")
        skiprows += 1

        # Parse only tables listed increasing lower level i, e.g. `FE/II/24may96/osc_nahar.dat`
        nrows = int(header["Number of transitions"])
        config = {
            "header": None,
            "index_col": False,
            "sep": "\s*\|\s*|-?\s+-?\s*|(?<=[^ED\s])-(?=[^\s])",
            "skiprows": skiprows,
            "nrows": nrows,
            "engine": "python",
        }

        columns = [
            "label_lower",
            "label_upper",
            "f",
            "A",
            "Lam(A)",
            "i",
            "j",
            "Lam(obs)",
            "% Acc",
        ]

        try:
            df = pd.read_csv(fname, **config)

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            logger.warning(f"Empty table: `{fname}`.")

        if df.shape[1] == 9:
            df.columns = columns

        # These files have 9-column, but the current regex produces 10 columns
        elif df.shape[1] == 10:
            df.columns = columns + ["?"]
            df = df.drop(columns=["?"])

        elif df.shape[1] == 8:
            df.columns = columns[:7] + ["#"]
            df = df.drop(columns=["#"])
            df["Lam(obs)"] = np.nan
            df["% Acc"] = np.nan

        else:
            logger.warning(f"Unknown column format: `{fname}`.")

        if df.shape[0] > 0 and "D" in str(df["f"][0]):
            df["f"] = df["f"].map(to_float)
            df["A"] = df["A"].map(to_float)

        self.base = df
        self.header = header


class CMFGENCollisionalStrengthsParser(BaseParser):
    """
    Description
    ----------
    base : pandas.DataFrame
    header : dict

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    def load(self, fname):
        header = parse_header(fname)
        skiprows, _ = find_row(fname, "ransition\T")
        config = {
            "header": None,
            "index_col": False,
            "sep": "\s*-?\s+-?|(?<=[^edED])-|(?<=[FDPS]e)-",
            "skiprows": skiprows,
            "engine": "python",
        }

        # FIXME: expensive solution for two files containing more than one
        # table: `ARG/III/19nov07/col_ariii` & `HE/II/5dec96/he2col.dat`
        end, _ = find_row(fname, "Johnson values:", "dln_OMEGA_dlnT", how="OR")

        if end is not None:
            config["nrows"] = end - config["skiprows"] - 2

        try:
            _, columns = find_row(fname, "ransition\T")
            columns = columns.split()

            # NOTE: Comment next line when trying new regexes
            columns = [
                np.format_float_scientific(to_float(x) * 1e04, precision=4)
                for x in columns[1:]
            ]
            config["names"] = ["label_lower", "label_upper"] + columns

        # FIXME: some files have no column names nor header
        except AttributeError:
            logger.warning(f"Unknown column format: `{fname}`.")

        try:
            df = pd.read_csv(fname, **config)
            for c in df.columns[2:]:  # This is done column-wise on purpose
                try:
                    df[c] = df[c].astype("float64")

                except ValueError:
                    df[c] = df[c].map(to_float).astype("float64")

        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
            logger.warning(f"Empty table: `{fname}`.")

        self.base = df
        self.header = header


class CMFGENPhoCrossSectionsParser(BaseParser):
    """
    Description
    ----------
    base : list of pandas.DataFrame
    header : dict

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    def _table_gen(self, f):
        """Yields a cross section table for a single energy level target.

        Parameters
        ----------
        f : file buffer

        Yields
        -------
        pd.DataFrame
            DataFrame with attached metadata.
        """

        label_key = "Configuration name"
        type_key = "Type of cross-section"
        num_key = "Number of cross-section points"

        data = []
        meta = {}

        for line in f:

            try:
                value = line.split()[0]

            except IndexError:
                continue

            if f"!{label_key}" in line:
                meta[f"{label_key}"] = value

            if f"!{type_key}" in line:
                meta[f"{type_key}"] = int(value)

            if f"!{num_key}" in line:
                n_points = int(value)
                for i in range(n_points):

                    values = f.readline().split()
                    if len(values) == 8:  # Verner & Yakolev (1995) ground state fits
                        data.append(
                            list(map(int, values[:2])) + list(map(to_float, values[2:]))
                        )

                        if i == n_points / len(values) - 1:
                            break

                    else:
                        data.append(map(to_float, values))

                meta[f"{num_key}"] = n_points
                break

        arr = np.array(data)
        yield arr, meta

    def load(self, fname):

        data = []
        header = parse_header(fname)
        with open_cmfgen_file(fname) as f:
            while True:

                arr, meta_ = next(self._table_gen(f), None)
                df = pd.DataFrame.from_records(arr)

                if df.empty:
                    break

                elif df.shape[1] == 2:
                    columns = ["energy", "sigma"]

                elif df.shape[1] == 1:
                    columns = ["fit_coeff"]

                elif df.shape[1] == 8:  # Verner & Yakolev (1995) ground state fits
                    columns = ["n", "l", "E", "E_0", "sigma_0", "y(a)", "P", "y(w)"]

                else:
                    logger.warning(f"Unknown column format: `{fname}`.")

                df.columns = columns
                df.attrs = meta_
                data.append(df)

        self.base = data
        self.header = header


class CMFGENHydLParser(BaseParser):
    """
    Parser for the CMFGEN hydrogen photoionization cross sections.

    Attributes
    ----------
    base : pandas.DataFrame, dtype float
        Photoionization cross section table for hydrogen. Values are the
        common logarithm (i.e. base 10) of the cross section in units cm^2.
        Indexed by the principal quantum number n and orbital quantum
        number l.
        Columns are the frequencies for the cross sections. Given in units
        of the threshold frequency for photoionization.
    header : dict

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    nu_ratio_key = "L_DEL_U"

    def load(self, fname):
        header = parse_header(fname)
        self.header = header
        self.max_l = self.get_max_l()

        self.num_xsect_nus = int(header["Number of values per cross-section"])
        nu_ratio = 10 ** float(header[self.nu_ratio_key])
        nus = np.power(
            nu_ratio, np.arange(self.num_xsect_nus)
        )  # in units of the threshold frequency

        skiprows, _ = find_row(fname, self.nu_ratio_key)
        skiprows += 1

        data = []
        indexes = []
        with open(fname, mode="r") as f:
            for i in range(skiprows):
                f.readline()
            while True:
                n, l, log10x_sect = next(self._table_gen(f), None)
                indexes.append((n, l))
                data.append(log10x_sect)
                if l == self.max_l:
                    break

        index = pd.MultiIndex.from_tuples(indexes, names=["n", "l"])
        self.base = pd.DataFrame(data, index=index, columns=nus)
        self.base.columns.name = "nu / nu_0"

    def _table_gen(self, f):
        """Yields a logarithmic cross section table for a hydrogen level.

        Parameters
        ----------
        f : file buffer

        Yields
        -------
        int
            Principal quantum number n.
        int
            Principal quantum number l.
        numpy.ndarray, dtype float
            Photoionization cross section table. Values are the common
            logarithm (i.e. base 10) of the cross section in units cm^2.
        """
        line = f.readline()
        n, l, num_entries = self.parse_table_header_line(line)
        assert num_entries == self.num_xsect_nus

        log10_xsect = []
        while True:
            line = f.readline()
            if not line.strip():  # This is the end of the current table
                break
            log10_xsect += [float(entry) for entry in line.split()]

        log10_xsect = np.array(log10_xsect)
        assert len(log10_xsect) == self.num_xsect_nus

        yield n, l, log10_xsect

    @staticmethod
    def parse_table_header_line(line):
        return [int(entry) for entry in line.split()]

    def get_max_l(self):
        return int(self.header["Maximum principal quantum number"]) - 1


class CMFGENHydGauntBfParser(CMFGENHydLParser):
    """
    Parser for the CMFGEN hydrogen bound-free gaunt factors.

    Attributes
    ----------
    base : pandas.DataFrame, dtype float
        Bound-free gaunt factors for hydrogen.
        Indexed by the principal quantum number n.
        Columns are the frequencies for the gaunt factors. Given in units of
        the threshold frequency for photoionization.
    header : dict

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    nu_ratio_key = "N_DEL_U"

    @staticmethod
    def parse_table_header_line(line):
        line_split = [int(entry) for entry in line.split()]
        n, l, num_entries = (
            line_split[0],
            line_split[0],
            line_split[1],
        )  # use n as mock l value
        return n, l, num_entries

    def load(self, fname):
        super().load(fname)
        self.base.index = self.base.index.droplevel("l")

    def get_max_l(self):
        return int(self.header["Maximum principal quantum number"])


class CMFGENReader:
    """
    Class for extracting levels and lines from CMFGEN.

    Mimics the GFALLReader class.

    Attributes
    ----------
    levels : DataFrame
    lines : DataFrame
    collisions : DataFrame

    """

    def __init__(
        self,
        data,
        priority=10,
        collisions=False,
        temperature_grid=None,
        drop_mismatched_labels=False,
        version=None,
    ):
        """
        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with
            keys `levels` and `lines`.
        collisions : bool
            Option to calculate collisional data, by default False.
        priority: int, optional
            Priority of the current data source, by default 10.
        temperature_grid : array/list of numbers, optional
            Temperatures to have in the collision dataframe. The collision dataframe
            will have all the temperatures from the CMFGEN dataset by default.

        """
        self.priority = priority
        self.ions = list(data.keys())
        self._get_levels_lines(data)
        if collisions:
            self.collisions, self.collisional_metadata = self._get_collisions(
                data,
                temperature_grid=temperature_grid,
                drop_mismatched_labels=drop_mismatched_labels,
            )
        self.version = version

    @classmethod
    def from_config(
        cls,
        ions,
        atomic_path,
        priority=10,
        ionization_energies=False,
        cross_sections=False,
        config_yaml=None,
        collisions=False,
        temperature_grid=None,
        drop_mismatched_labels=False,
    ):
        ATOMIC_PATH = pathlib.Path(atomic_path)
        if config_yaml is not None:
            YAML_PATH = pathlib.Path(config_yaml).as_posix()

        else:
            YAML_PATH = (
                pathlib.Path(CARSUS_PATH[0])
                .joinpath("data", "cmfgen_config.yml")
                .as_posix()
            )

        data = {}
        with open(YAML_PATH) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            version = conf["version"]
            ions = parse_selected_species(ions)

            if cross_sections and (1, 0) not in ions:
                logger.warning("Selecting H 0 required to ingest cross-sections.")
                ions.insert(0, (1, 0))

            for ion in ions:
                symbol = convert_atomic_number2symbol(ion[0])

                try:
                    ion_keys = conf["atom"][symbol]["ion_charge"][ion[1]]
                    BASE_PATH = ATOMIC_PATH.joinpath(
                        CMFGEN_ATOM_DICT[symbol],
                        roman.toRoman(ion[1] + 1),
                        ion_keys["date"],
                    )

                    logger.info(f"Configuration schema found for {symbol} {ion[1]}.")

                except KeyError:
                    raise KeyError(f'Configuration schema missing for {symbol} {ion[1]}.'
                                    'Please check the CMFGEN configuration file: carsus/data/cmfgen_config.yml'
                    )

                osc_fname = BASE_PATH.joinpath(ion_keys["osc"]).as_posix()
                col_fname = BASE_PATH.joinpath(ion_keys["col"]).as_posix()

                data[ion] = {}

                lvl_parser = CMFGENEnergyLevelsParser(osc_fname)
                lns_parser = CMFGENOscillatorStrengthsParser(osc_fname)
                data[ion]["levels"] = lvl_parser.base.copy()
                data[ion]["lines"] = lns_parser.base.copy()

                if collisions:
                    col_parser = CMFGENCollisionalStrengthsParser(col_fname)
                    data[ion]["collisions"] = col_parser.base.copy()

                if ionization_energies:
                    data[ion]["ionization_energy"] = float(
                        lvl_parser.header["Ionization energy"]
                    )

                if cross_sections:
                    pho_flist = []
                    try:
                        for j, k in enumerate(ion_keys["pho"]):
                            pho_fname = BASE_PATH.joinpath(
                                ion_keys["pho"][j]
                            ).as_posix()
                            pho_flist.append(pho_fname)

                    except KeyError:
                        logger.warning(f"No `pho` data for {symbol} {ion[1]}.")

                    data[ion]["cross_sections"] = []
                    for l in pho_flist:
                        pho_parser = CMFGENPhoCrossSectionsParser(l)
                        data[ion]["cross_sections"].append(pho_parser.base)

                    if ion == (1, 0):
                        hyd_fname = BASE_PATH.joinpath("hyd_l_data.dat").as_posix()
                        gbf_fname = BASE_PATH.joinpath("gbf_n_data.dat").as_posix()

                        hyd_parser = CMFGENHydLParser(hyd_fname)
                        gbf_parser = CMFGENHydGauntBfParser(gbf_fname)

                        data[ion]["hyd"] = hyd_parser.base
                        data[ion]["gbf"] = gbf_parser.base

        return cls(data, priority, collisions, temperature_grid, drop_mismatched_labels, version)

    @staticmethod
    def cross_sections_squeeze(
        reader_phixs,
        ion_levels,
        hyd_phixs_energy_grid_ryd,
        hyd_phixs,
        hyd_gaunt_energy_grid_ryd,
        hyd_gaunt_factor,
    ):
        """
        Makes a single, uniform table of cross-sections from individual DataFrames
        based on cross-section types and their respective papers.
        """

        phixs_table_list = []
        n_targets = len(reader_phixs)

        for i in range(n_targets):

            target = reader_phixs[i]
            lower_level_label = target.attrs["Configuration name"]
            cross_section_type = target.attrs["Type of cross-section"]

            # Remove the "[J]" term from J-splitted levels labels
            ion_levels["label"] = ion_levels["label"].str.rstrip("]")
            ion_levels["label"] = ion_levels["label"].str.split("[", expand=True)

            try:
                match = ion_levels.set_index("label").loc[[lower_level_label]]

            except KeyError:
                logger.warning(f"Level not found: '{lower_level_label}'.")
                continue

            lambda_angstrom = match["Lam(A)"].tolist()
            level_number = (match["ID"] - 1).tolist()
            
            # match is > 1 just for J-splitted levels
            for j in range(len(match)):
                threshold_energy_ryd = (
                    HC_IN_EV_ANGSTROM / lambda_angstrom[j] / RYD_TO_EV
                )

                if cross_section_type == CrossSectionType.CONSTANT_ZERO:
                    phixs_table = get_null_phixs_table(threshold_energy_ryd)

                elif cross_section_type in [
                    CrossSectionType.POINTS_TABLE,
                    CrossSectionType.OPACITY_PROJECT_SC,
                    CrossSectionType.OPACITY_PROJECT_SC_SM,
                ]:

                    diff = target["energy"].diff().dropna()
                    assert (diff >= 0).all()

                    energy = (target["energy"] * threshold_energy_ryd).values
                    sigma = target["sigma"].values
                    phixs_table = np.column_stack((energy, sigma))

                elif cross_section_type in [
                    CrossSectionType.SEATON_FITS,
                    CrossSectionType.SEATON_FITS_OFFSET,
                ]:

                    fit_coeff_list = target["fit_coeff"].to_list()

                    if len(fit_coeff_list) not in [1, 3, 4]:
                        logger.warning(
                            f"Inconsistent number of fit coefficients for '{lower_level_label}'."
                        )
                        continue

                    if len(fit_coeff_list) == 1 and fit_coeff_list[0] == 0.0:
                        phixs_table = get_null_phixs_table(threshold_energy_ryd)

                    else:
                        phixs_table = get_seaton_phixs_table(
                            threshold_energy_ryd, *fit_coeff_list
                        )

                elif cross_section_type == CrossSectionType.HYDROGENIC_PURE_N_LEVEL:
                    fit_coeff_list = target["fit_coeff"].to_list()

                    if len(fit_coeff_list) != 2:
                        logger.warning(
                            f"Inconsistent number of fit coefficients for '{lower_level_label}'."
                        )
                        continue

                    scale, n = fit_coeff_list
                    phixs_table = scale * get_hydrogenic_n_phixs_table(
                        hyd_gaunt_energy_grid_ryd,
                        hyd_gaunt_factor,
                        threshold_energy_ryd,
                        n,
                    )

                elif cross_section_type in [
                    CrossSectionType.HYDROGENIC_SPLIT_L,
                    CrossSectionType.HYDROGENIC_SPLIT_L_OFFSET,
                ]:

                    fit_coeff_list = target["fit_coeff"].to_list()
                    fit_coeff_list[0:3] = [int(x) for x in fit_coeff_list[0:3]]

                    if len(fit_coeff_list) not in [3, 4]:
                        logger.warning(
                            f"Inconsistent number of fit coefficients for '{lower_level_label}'."
                        )
                        continue

                    phixs_table = get_hydrogenic_nl_phixs_table(
                        hyd_phixs_energy_grid_ryd,
                        hyd_phixs,
                        threshold_energy_ryd,
                        *fit_coeff_list,
                    )

                elif cross_section_type == CrossSectionType.OPACITY_PROJECT_FITS:
                    fit_coeff_list = target["fit_coeff"].to_list()

                    if len(fit_coeff_list) != 5:
                        logger.warning(
                            f"Inconsistent number of fit coefficients for '{lower_level_label}'."
                        )
                        continue

                    phixs_table = get_opproject_phixs_table(
                        threshold_energy_ryd, *fit_coeff_list
                    )

                elif cross_section_type == CrossSectionType.HUMMER_HEI_FITS:
                    fit_coeff_list = target["fit_coeff"].to_list()

                    if len(fit_coeff_list) != 8:
                        logger.warning(
                            f"Inconsistent number of fit coefficients for '{lower_level_label}'."
                        )
                        continue

                    phixs_table = get_hummer_phixs_table(
                        threshold_energy_ryd, *fit_coeff_list
                    )

                elif cross_section_type == CrossSectionType.VERNER_YAKOLEV_GS_FITS:
                    fit_coeff_table = target

                    if fit_coeff_table.shape[1] != 8:
                        logger.warning(
                            f"Inconsistent number of fit coefficients for '{lower_level_label}'."
                        )
                        continue

                    phixs_table = get_vy95_phixs_table(
                        threshold_energy_ryd, fit_coeff_table
                    )

                elif cross_section_type == CrossSectionType.LEIBOWITZ_CIV_FITS:
                    fit_coeff_list = target["fit_coeff"].tolist()

                    if len(fit_coeff_list) != 6:
                        logger.warning(
                            f"Inconsistent number of fit coefficients for '{lower_level_label}'."
                        )
                        continue

                    try:
                        phixs_table = get_leibowitz_phixs_table(
                            threshold_energy_ryd, *fit_coeff_list
                        )

                    except NotImplementedError:
                        logger.warning(
                            f"Leibowitz's cross-section type 4 not implemented yet."
                        )
                        phixs_table = get_null_phixs_table(threshold_energy_ryd)

                else:
                    logger.warning(
                        f"Unknown cross-section type {cross_section_type} for configuration '{lower_level_label}'."
                    )
                    continue

                df = pd.DataFrame(phixs_table, columns=["energy", "sigma"])
                df["level_index"] = level_number[j]
                
                phixs_table_list.append(df)

        ion_phixs_table = pd.concat(phixs_table_list)

        return ion_phixs_table

    def _get_levels_lines(self, data):
        """
        Generates `levels`, `lines` and (optionally) `ionization_energies` and
        `collisions` DataFrames.
        """

        lvl_list = []
        lns_list = []
        ioz_list = []
        pxs_list = []

        for ion, reader in data.items():
            atomic_number = ion[0]
            ion_charge = ion[1]

            symbol = convert_atomic_number2symbol(ion[0])
            logger.info(f"Loading atomic data for {symbol} {ion[1]}.")

            lvl = reader["levels"].copy()
            # some ID's have negative values (theoretical?)
            lvl.loc[lvl["ID"] < 0, "method"] = "theor"
            lvl.loc[lvl["ID"] > 0, "method"] = "meas"
            lvl["ID"] = np.abs(lvl["ID"])
            lvl_id = lvl.set_index("ID")

            lvl["atomic_number"] = atomic_number
            lvl["ion_charge"] = ion_charge
            lvl_list.append(lvl)

            lns = reader["lines"].copy()
            lns = lns.set_index(["i", "j"])
            lns["energy_lower"] = lvl_id["E(cm^-1)"].reindex(lns.index, level=0).values
            lns["energy_upper"] = lvl_id["E(cm^-1)"].reindex(lns.index, level=1).values
            lns["g_lower"] = lvl_id["g"].reindex(lns.index, level=0).values
            lns["g_upper"] = lvl_id["g"].reindex(lns.index, level=1).values
            lns["j_lower"] = (lns["g_lower"] - 1) / 2
            lns["j_upper"] = (lns["g_upper"] - 1) / 2
            lns["atomic_number"] = atomic_number
            lns["ion_charge"] = ion_charge
            lns = lns.reset_index()
            lns_list.append(lns)

            if "ionization_energy" in reader.keys():
                ioz_list.append(
                    {
                        "atomic_number": ion[0],
                        "ion_charge": ion[1],
                        "ionization_energy": reader["ionization_energy"],
                    }
                )

            if "cross_sections" in reader.keys():
                if ion == (1, 0):

                    # Extracted from the header of HYD files since the `data` object
                    # passed to this method does not contain header information.

                    n_levels = 30
                    l_points, l_start_u, l_del_u = 97, 0.0, 0.041392685
                    n_points, n_start_u, n_del_u = 145, 0.0, 0.041392685

                    hyd = reader["hyd"].apply(lambda x: 10 ** (8 + x))
                    gbf = reader["gbf"]

                    hyd_phixs, hyd_phixs_energy_grid_ryd = {}, {}
                    hyd_gaunt_factor, hyd_gaunt_energy_grid_ryd = {}, {}

                    for n in range(1, n_levels + 1):

                        lambda_angstrom = lvl.loc[n - 1, "Lam(A)"]
                        e_threshold_ev = HC_IN_EV_ANGSTROM / lambda_angstrom

                        hyd_gaunt_energy_grid_ryd[n] = [
                            e_threshold_ev / RYD_TO_EV * 10 ** (n_start_u + n_del_u * i)
                            for i in range(n_points)
                        ]
                        hyd_gaunt_factor[n] = gbf.loc[n].tolist()

                        for l in range(0, n):
                            hyd_phixs_energy_grid_ryd[(n, l)] = [
                                e_threshold_ev
                                / RYD_TO_EV
                                * 10 ** (l_start_u + l_del_u * i)
                                for i in range(l_points)
                            ]
                            hyd_phixs[(n, l)] = hyd.loc[(n, l)].tolist()

                pxs = self.cross_sections_squeeze(
                    reader["cross_sections"][0],
                    lvl.copy(),
                    hyd_phixs_energy_grid_ryd,
                    hyd_phixs,
                    hyd_gaunt_energy_grid_ryd,
                    hyd_gaunt_factor,
                )
                pxs["atomic_number"] = ion[0]
                pxs["ion_charge"] = ion[1]
                pxs_list.append(pxs)

        levels = pd.concat(lvl_list)
        levels["priority"] = self.priority
        levels = levels.reset_index(drop=False)
        levels = levels.rename(
            columns={"label": "label", "E(cm^-1)": "energy", "index": "level_index"}
        )
        levels["j"] = (levels["g"] - 1) / 2
        levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
        levels = levels[["energy", "j", "label", "method", "priority"]]

        lines = pd.concat(lns_list)
        lines = lines.rename(columns={"Lam(A)": "wavelength"})
        lines["wavelength"] = u.Quantity(lines["wavelength"], "AA").to("nm").value
        lines["level_index_lower"] = lines["i"] - 1
        lines["level_index_upper"] = lines["j"] - 1
        lines["gf"] = lines["f"] * lines["g_lower"]
        lines = lines.set_index(
            ["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"]
        )
        lines = lines[
            ["energy_lower", "energy_upper", "gf", "j_lower", "j_upper", "wavelength"]
        ]

        if "ionization_energy" in reader.keys():
            ionization_energies = pd.DataFrame.from_records(ioz_list)
            ionization_energies["ionization_energy"] = (
                (ionization_energies["ionization_energy"].values / u.cm)
                .to("eV", equivalencies=u.spectral())
                .value
            )
            ionization_energies = ionization_energies.set_index(
                ["atomic_number", "ion_charge"]
            ).squeeze()
            self.ionization_energies = ionization_energies

        if "cross_sections" in reader.keys():
            cross_sections = pd.concat(pxs_list)
            cross_sections = cross_sections.set_index(
                ["atomic_number", "ion_charge", "level_index"]
            )
            self.cross_sections = cross_sections.sort_index()

        self.levels = levels
        self.lines = lines

        return

    def _get_collisions(
        self, data, temperature_grid=None, drop_mismatched_labels=False
    ):
        """
        Generate the `collisions` DataFrame.

        Parameters
        ----------
        data : dict
           Dictionary containing one dictionary per species with
           keys `levels` and `lines`.
        temperature_grid : array/list of numbers, optional
            Temperatures to have in the collision dataframe. The collision dataframe
            will have all the temperatures from the CMFGEN dataset by default.
        """
        col_list, t_grid = [], []
        levels_combine = self.levels.copy().reset_index()
        label_ind_mapping = {
            label: index
            for label, index in zip(levels_combine.label, levels_combine.level_index)
        }

        for ion, data_dict in data.items():
            levels = data_dict["levels"].copy()
            collisions = data_dict["collisions"].copy()

            label_g_mapping = {label: g for label, g in zip(levels.label, levels.g)}
            missing_labels = set()

            gi, lower_level_index, upper_level_index = [], [], []

            for ll, ul in zip(collisions.label_lower, collisions.label_upper):
                if ll in label_ind_mapping:
                    lower_level_index.append(label_ind_mapping[ll])
                else:
                    if not drop_mismatched_labels:
                        raise KeyError(
                            f"Label {ll} for ion {ion} could not be mapped. "
                            "Please check the atomic data files."
                        )
                    missing_labels.add(ll)
                    lower_level_index.append(np.nan)

                if ll in label_g_mapping:
                    gi.append(label_g_mapping[ll])
                else:
                    if not drop_mismatched_labels:
                        raise KeyError(
                            f"Label {ll} for ion {ion} could not be mapped. "
                            "Please check the atomic data files."
                        )
                    missing_labels.add(ll)
                    gi.append(np.nan)

                if ul in label_ind_mapping:
                    upper_level_index.append(label_ind_mapping[ul])
                else:
                    if ul != "I":
                        if not drop_mismatched_labels:
                            raise KeyError(
                                f"Label {ul} for ion {ion} could not be mapped. "
                                "Please check the atomic data files."
                            )
                        missing_labels.add(ul)
                    else:
                        logger.info("Dropping collisional ionization data.")

                    upper_level_index.append(np.nan)

            if missing_labels:
                logger.info(
                    f"Entries having label(s): {', '.join(missing_labels)} will be dropped for ion: {ion}."
                )

            collisions["level_number_lower"] = lower_level_index
            collisions["level_number_upper"] = upper_level_index
            collisions["gi"] = gi

            collisions = collisions.dropna(
                subset=["level_number_lower", "level_number_upper"]
            )

            collisions["atomic_number"] = ion[0]
            collisions["ion_number"] = ion[1]
            collisions = collisions.drop(columns=["label_lower", "label_upper"])

            collisions = collisions.astype(
                {
                    "level_number_upper": int,
                    "level_number_lower": int,
                }
            )

            collisions = collisions.set_index(
                [
                    "atomic_number",
                    "ion_number",
                    "level_number_lower",
                    "level_number_upper",
                ]
            )
            # divide the dataframe by gi and remove the column
            collisions = collisions.iloc[:, :-1].div(collisions.gi, axis=0)
            collisions.columns = collisions.columns.astype(float)

            t_grid.extend(collisions.columns)
            col_list.append(collisions)

        if temperature_grid is None:
            temperature_grid = sorted(list(set(t_grid)))

        col_interp = []
        for ion_col_data, ion in zip(col_list, data.keys()):
            ion_col_data_columns = ion_col_data.columns
            interpolated_values = [
                np.interp(
                    x=temperature_grid, xp=ion_col_data_columns, fp=ion_col_data_entry
                )
                for ion_col_data_entry in ion_col_data.values
            ]

            nans_replaced = len(
                np.where(
                    np.logical_or(
                        temperature_grid < ion_col_data_columns[0],
                        temperature_grid > ion_col_data_columns[-1],
                    )
                )[0]
            )
            if nans_replaced:
                logger.info(
                    f"Filling in {nans_replaced} "
                    f"values for ion {ion} that are outside the tabulated "
                    "temperature range using the last tabulated value."
                )

            ion_interp = pd.DataFrame(
                interpolated_values, index=ion_col_data.index, columns=temperature_grid
            )
            col_interp.append(ion_interp)

        collisions = pd.concat(col_interp)
        collisions = collisions[sorted(collisions.columns)]

        metadata = pd.Series(
            {
                "temperatures": collisions.columns.astype(int).values,
                "dataset": ["cmfgen"],
                "info": "The dataframe values are thermally-averaged effective collision "
                "strengths divided by the statistical weights of the lower levels.",
            }
        )
        collisions.columns = range(collisions.shape[1])

        return collisions, metadata
