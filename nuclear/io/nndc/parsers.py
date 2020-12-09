from abc import ABCMeta

import pandas as pd
from astropy import units as u
from uncertainties import ufloat_fromstr
import numpy as np


class BaseParser(metaclass=ABCMeta):
    @staticmethod
    def _convert_html_to_df(html_table, column_names):
        df = pd.read_html(str(html_table))[0].iloc[1:]
        df.columns = column_names
        if "type" in column_names:
            df.type[df.type.isnull()] = ""
        return df

    @staticmethod
    def _parse_uncertainties(unc_raw_str, split_unc_symbol='%'):
        unc_raw_str = unc_raw_str.lower()

        value_unc_pair = [item.strip()
                      for item in unc_raw_str.split(split_unc_symbol)]

        if len(value_unc_pair) == 1:  # if no uncertainty given
            return float(value_unc_pair[0]), np.nan
        else:
            value, unc = value_unc_pair
        if unc == "":  # if uncertainty_str is empty
            unc = np.nan
        if "e" in value:
            exp_pos = value.find("e")
            unc_str = value[:exp_pos] + f"({unc})" + value[exp_pos:]
        else:
            unc_str = value + f"({unc})"
        parsed_uncertainty = ufloat_fromstr(unc_str)
        return parsed_uncertainty.nominal_value, parsed_uncertainty.std_dev

    def _sanititze_table(self, df):
        df.dropna(inplace=True)

        if "intensity" in df.columns:
            intensity_raw = df["intensity"].apply(
                BaseParser._parse_uncertainties)
            df.loc[:, "intensity"] = [item[0] for item in intensity_raw]
            df.loc[:, "intensity_unc"] = [item[1]
                                          for item in intensity_raw]
        if "energy" in df.columns:
            energy = df["energy"].apply(
                BaseParser._parse_uncertainties, split_unc_symbol=' ')
            df.loc[:, "energy"] = [item[0] for item in energy]
            df.loc[:, "energy_unc"] = [item[1] for item in energy]

        if "end_point_energy" in df.columns:
            end_point_energy = df["end_point_energy"].apply(
                BaseParser._parse_uncertainties, split_unc_symbol=' ')
            df.loc[:, "end_point_energy"] = [
                item[0] for item in end_point_energy]
            df.loc[:, "end_point_energy_unc"] = [
                item[1] for item in end_point_energy]

        if "dose" in df.columns:
            del df["dose"]
        df['heading'] = self.html_name
        return df

    def _default_parse(self, html_table):
        df = self._convert_html_to_df(html_table, self.columns)
        df = self._sanititze_table(df)
        df['type'] = self.type
        return df

    def parse(self, html_table):
        return self._default_parse(html_table)


class ElectronTableParser(BaseParser):
    html_name = "Electrons"
    type = "e-"
    columns = ["type", "energy", "intensity", "dose"]



class BetaPlusTableParser(BaseParser):
    html_name = "Beta+"
    type = "e+"
    columns = ["energy", "end_point_energy", "intensity", "dose"]


class BetaMinusTableParser(BaseParser):
    html_name = "Beta-"
    type = "e-"
    columns = ["energy", "end_point_energy", "intensity", "dose"]


class XGammaRayParser(BaseParser):
    html_name = "Gamma and X-ray radiation"
    columns = ["type", "energy", "intensity", "dose"]

    def parse(self, html_table):
        df = self._convert_html_to_df(html_table, self.columns)
        df = self._sanititze_table(df)

        x_ray_mask = df.type.str.startswith("XR")

        df['type'] = 'x_rays'
        df.loc[~x_ray_mask, 'type'] = 'gamma_rays'
        return df


decay_radiation_parsers = {
    item.html_name: item() for item in BaseParser.__subclasses__()
}
