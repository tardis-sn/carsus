import re
import logging
import numpy as np
import pandas as pd
from carsus.util import parse_selected_species
from carsus.io.util import read_from_buffer


CARSUS_DATA_GFALL_URL = "https://github.com/tardis-sn/carsus-data-kurucz/raw/main/linelists/gfall/gfall.dat?raw=true"
GFALL_AIR_THRESHOLD = 200  # [nm], wavelengths above this value are given in air

logger = logging.getLogger(__name__)


class GFALLReader(object):
    """
    Class for extracting lines and levels data from kurucz gfall files

    Attributes
    ----------
    fname: path to gfall.dat

    Methods
    --------
    gfall_raw:
        Return pandas DataFrame representation of gfall

    """

    gfall_fortran_format = (
        "F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,"
        "A10,F6.2,F6.2,F6.2,A4,I2,I2,I3,F6.3,I3,F6.3,I5,I5,"
        "1X,I1,A1,1X,I1,A1,I1,A3,I5,I5,I6"
    )

    gfall_columns = [
        "wavelength",
        "loggf",
        "element_code",
        "e_first",
        "j_first",
        "blank1",
        "label_first",
        "e_second",
        "j_second",
        "blank2",
        "label_second",
        "log_gamma_rad",
        "log_gamma_stark",
        "log_gamma_vderwaals",
        "ref",
        "nlte_level_no_first",
        "nlte_level_no_second",
        "isotope",
        "log_f_hyperfine",
        "isotope2",
        "log_iso_abundance",
        "hyper_shift_first",
        "hyper_shift_second",
        "blank3",
        "hyperfine_f_first",
        "hyperfine_note_first",
        "blank4",
        "hyperfine_f_second",
        "hyperfine_note_second",
        "line_strength_class",
        "line_code",
        "lande_g_first",
        "lande_g_second",
        "isotopic_shift",
    ]

    default_unique_level_identifier = ["energy", "j"]

    def __init__(
        self, ions=None, fname=None, unique_level_identifier=None, priority=10
    ):
        """
        Parameters
        ----------
        fname: str
            Path to the gfall file (http or local file).

        ions: str, optional
            Ions to extract, by default None.

        unique_level_identifier: list
            List of attributes to identify unique levels from. Will always use
            atomic_number and ion charge in addition.

        priority: int, optional
            Priority of the current data source.
        """

        if fname is None:
            self.fname = CARSUS_DATA_GFALL_URL
        else:
            self.fname = fname

        self.priority = priority

        if ions is not None:
            self.ions = parse_selected_species(ions)

        else:
            self.ions = None

        self._gfall_raw = None
        self._gfall = None
        self._levels = None
        self._lines = None
        if unique_level_identifier is None:
            logger.warning(
                "A specific combination to identify unique levels from "
                "GFALL data has not been given. Defaulting to "
                '["energy", "j"].'
            )
            self.unique_level_identifier = self.default_unique_level_identifier
        else:
            self.unique_level_identifier = unique_level_identifier

    @property
    def gfall_raw(self):
        if self._gfall_raw is None:
            self._gfall_raw, self.version = self.read_gfall_raw()
        return self._gfall_raw

    @property
    def gfall(self):
        if self._gfall is None:
            self._gfall = self.parse_gfall()
        return self._gfall

    @property
    def levels(self):
        if self._levels is None:
            self._levels = self.extract_levels()
        return self._levels

    @property
    def lines(self):
        if self._lines is None:
            self._lines = self.extract_lines()
        return self._lines

    def read_gfall_raw(self, fname=None):
        """
        Reading in a normal gfall.dat

        Parameters
        ----------
        fname: ~str
            path to gfall.dat

        Returns
        -------
            pandas.DataFrame
                pandas Dataframe represenation of gfall

            str
                MD5 checksum
        """

        if fname is None:
            fname = self.fname

        logger.info(f"Parsing GFALL from: {fname}")

        # FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
        # 3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3,2I5,I6)

        number_match = re.compile(r"\d+(\.\d+)?")
        type_match = re.compile(r"[FIXA]")
        type_dict = {"F": np.float64, "I": np.int64, "X": str, "A": str}
        field_types = tuple(
            [
                type_dict[item]
                for item in number_match.sub("", self.gfall_fortran_format).split(",")
            ]
        )

        field_widths = type_match.sub("", self.gfall_fortran_format)
        field_widths = map(int, re.sub(r"\.\d+", "", field_widths).split(","))

        field_type_dict = {
            col: dtype for col, dtype in zip(self.gfall_columns, field_types)
        }

        buffer, checksum = read_from_buffer(self.fname)
        gfall = pd.read_fwf(
            buffer,
            widths=field_widths,
            skip_blank_lines=True,
            names=self.gfall_columns,
            dtypes=field_type_dict,
        )

        # remove empty lines
        gfall = gfall[~gfall.isnull().all(axis=1)].reset_index(drop=True)
        return gfall, checksum

    def parse_gfall(self, gfall_raw=None):
        """
        Parse raw gfall DataFrame

        Parameters
        ----------
        gfall_raw: pandas.DataFrame

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """

        gfall = gfall_raw if gfall_raw is not None else self.gfall_raw.copy()
        gfall = gfall.rename(
            columns={"e_first": "energy_first", "e_second": "energy_second"}
        )
        double_columns = [
            item.replace("_first", "")
            for item in gfall.columns
            if item.endswith("first")
        ]

        # due to the fact that energy is stored in 1/cm
        order_lower_upper = gfall["energy_first"].abs() < gfall["energy_second"].abs()

        for column in double_columns:
            data = pd.concat(
                [
                    gfall["{0}_first".format(column)][order_lower_upper],
                    gfall["{0}_second".format(column)][~order_lower_upper],
                ]
            )

            gfall["{0}_lower".format(column)] = data

            data = pd.concat(
                [
                    gfall["{0}_first".format(column)][~order_lower_upper],
                    gfall["{0}_second".format(column)][order_lower_upper],
                ]
            )

            gfall["{0}_upper".format(column)] = data

            del gfall["{0}_first".format(column)]
            del gfall["{0}_second".format(column)]

        # Clean labels
        gfall["label_lower"] = gfall["label_lower"].str.strip()
        gfall["label_upper"] = gfall["label_upper"].str.strip()

        gfall["label_lower"] = gfall["label_lower"].str.replace(r"\s+", " ")
        gfall["label_upper"] = gfall["label_upper"].str.replace(r"\s+", " ")

        # Ignore lines with the labels "AVARAGE ENERGIES" and "CONTINUUM"
        ignored_labels = ["AVERAGE", "ENERGIES", "CONTINUUM"]
        gfall = gfall.loc[
            ~(
                (gfall["label_lower"].isin(ignored_labels))
                | (gfall["label_upper"].isin(ignored_labels))
            )
        ].copy()

        gfall["energy_lower_predicted"] = gfall["energy_lower"] < 0
        gfall["energy_lower"] = gfall["energy_lower"].abs()
        gfall["energy_upper_predicted"] = gfall["energy_upper"] < 0
        gfall["energy_upper"] = gfall["energy_upper"].abs()

        gfall["atomic_number"] = gfall.element_code.astype(int)
        gfall["ion_charge"] = (
            ((gfall.element_code.values - gfall.atomic_number.values) * 100)
            .round()
            .astype(int)
        )

        del gfall["element_code"]

        return gfall

    def extract_levels(self, gfall=None, selected_columns=None):
        """
        Extract levels from `gfall`. We first generate a concatenated DataFrame
        of all lower and upper levels. Then we drop the duplicate leves

        Parameters
        ----------
        gfall: pandas.DataFrame
        selected_columns: list
            list of which columns to select (optional - default=None which selects
            a default set of columns)

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """

        if gfall is None:
            gfall = self.gfall

        if selected_columns is None:
            selected_columns = [
                "atomic_number",
                "ion_charge",
                "energy",
                "j",
                "label",
                "theoretical",
            ]

        column_renames = {
            "energy_{0}": "energy",
            "j_{0}": "j",
            "label_{0}": "label",
            "energy_{0}_predicted": "theoretical",
        }

        e_lower_levels = gfall.rename(
            columns=dict(
                [(key.format("lower"), value) for key, value in column_renames.items()]
            )
        )

        e_upper_levels = gfall.rename(
            columns=dict(
                [(key.format("upper"), value) for key, value in column_renames.items()]
            )
        )

        levels = pd.concat(
            [e_lower_levels[selected_columns], e_upper_levels[selected_columns]]
        )
        unique_level_id = ["atomic_number", "ion_charge"] + self.unique_level_identifier

        levels.drop_duplicates(unique_level_id, inplace=True)
        levels = levels.sort_values(
            ["atomic_number", "ion_charge", "energy", "j", "label"]
        )

        levels["method"] = levels["theoretical"].apply(
            lambda x: "theor" if x else "meas"
        )  # Theoretical or measured
        levels.drop("theoretical", axis="columns", inplace=True)

        levels["level_index"] = (
            levels.groupby(["atomic_number", "ion_charge"])["j"]
            .transform(lambda x: np.arange(len(x), dtype=np.int64))
            .values
        )
        levels["level_index"] = levels["level_index"].astype(int)

        # ToDo: The commented block below does not work with all lines. Find a way to parse it.
        # levels[["configuration", "term"]] = levels["label"].str.split(expand=True)
        # levels["configuration"] = levels["configuration"].str.strip()
        # levels["term"] = levels["term"].s

        # TODO: move to a staticmethod
        if self.ions is not None:
            lvl_list = []
            for ion in self.ions:
                mask = (levels["atomic_number"] == ion[0]) & (
                    levels["ion_charge"] == ion[1]
                )
                lvl = levels[mask]
                lvl_list.append(lvl)

            levels = pd.concat(lvl_list, sort=True)

        levels.set_index(["atomic_number", "ion_charge", "level_index"], inplace=True)

        levels["priority"] = self.priority

        return levels

    def extract_lines(self, gfall=None, levels=None, selected_columns=None):
        """
        Extract lines from `gfall`

        Parameters
        ----------
        gfall: pandas.DataFrame
        selected_columns: list
            list of which columns to select (optional - default=None which selects
            a default set of columns)

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """
        if gfall is None:
            gfall = self.gfall

        if levels is None:
            levels = self.levels

        if selected_columns is None:
            selected_columns = ["atomic_number", "ion_charge"]
            selected_columns += [
                item + "_lower" for item in self.unique_level_identifier
            ]
            selected_columns += [
                item + "_upper" for item in self.unique_level_identifier
            ]
            selected_columns += ["wavelength", "loggf"]

        logger.info("Extracting line data: {0}.".format(", ".join(selected_columns)))
        unique_level_id = ["atomic_number", "ion_charge"] + self.unique_level_identifier
        levels_idx = levels.reset_index()
        levels_idx = levels_idx.set_index(unique_level_id)

        lines = gfall[selected_columns].copy()
        lines["gf"] = np.power(10, lines["loggf"])
        lines = lines.drop(["loggf"], axis="columns")

        # Assigning levels to lines
        levels_unique_idxed = self.levels.reset_index().set_index(
            ["atomic_number", "ion_charge"] + self.unique_level_identifier
        )

        lines_lower_unique_idx = ["atomic_number", "ion_charge"] + [
            item + "_lower" for item in self.unique_level_identifier
        ]
        lines_upper_unique_idx = ["atomic_number", "ion_charge"] + [
            item + "_upper" for item in self.unique_level_identifier
        ]
        lines_lower_idx = lines.set_index(lines_lower_unique_idx)
        lines_lower_idx["level_index_lower"] = levels_unique_idxed["level_index"]
        lines_upper_idx = lines_lower_idx.reset_index().set_index(
            lines_upper_unique_idx
        )
        lines_upper_idx["level_index_upper"] = levels_unique_idxed["level_index"]
        lines = lines_upper_idx.reset_index()

        # TODO: move to a staticmethod
        if self.ions is not None:
            lns_list = []
            for ion in self.ions:
                mask = (lines["atomic_number"] == ion[0]) & (
                    lines["ion_charge"] == ion[1]
                )
                lns = lines[mask]
                lns_list.append(lns)

            lines = pd.concat(lns_list, sort=True)

        lines["level_index_lower"] = lines["level_index_lower"].astype("int")
        lines["level_index_upper"] = lines["level_index_upper"].astype("int")
        lines.set_index(
            ["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"],
            inplace=True,
        )

        return lines

    def to_hdf(self, fname):
        """
        Parameters
        ----------
        fname : path
           Path to the HDF5 output file
        """
        with pd.HDFStore(fname, "w") as f:
            f.put("/gfall_raw", self.gfall_raw)
            f.put("/gfall", self.gfall)
