from abc import ABCMeta

import pandas as pd
from astropy import units as u
from uncertainties import ufloat_fromstr

class BaseParser(metaclass=ABCMeta):

    @staticmethod
    def _convert_html_to_df(html_table, column_names):
        df = pd.read_html(str(html_table))[0].iloc[1:]
        df.columns = column_names
        if 'type' in column_names:
            df.type[df.type.isnull()] = ''
        return df

    @staticmethod
    def _parse_uncertainties(unc_raw_str):
        unc_raw_str = unc_raw_str.lower()
        value, unc = [item.strip() for item in unc_raw_str.split('%')]
        if 'e' in value:
            exp_pos = value.find('e')
            unc_str = value[:exp_pos] + f'({unc})' + value[exp_pos:]
        else:
            unc_str = value + f'({unc})'

        parsed_uncertainty = ufloat_fromstr(unc_str)
        return parsed_uncertainty.nominal_value, parsed_uncertainty.std_dev

    @staticmethod
    def _sanititze_table(df):
        df = df.dropna()
        if 'energy' in df.columns:
            df.loc[:, 'energy'] = df.energy.apply(lambda x: u.Quantity(
                float(x.split()[0]), u.keV).to(u.erg).value)
        if 'end_point_energy' in df.columns:
            df.loc[:, 'end_point_energy'] = df.end_point_energy.apply(
                lambda x: u.Quantity(float(x.split()[0]), u.keV).to(
                    u.erg).value)

        if 'intensity' in df.columns:
            intensity_raw = df['intensity'].apply(
                BaseParser._parse_uncertainties)
            df.loc[:, 'intensity'] = [item[0] for item in intensity_raw]
            df.loc[:, 'intensity_unc'] = [item[1] for item in intensity_raw]

        if 'dose' in df.columns:
            del df['dose']

        return df

    def _default_parse(self, html_table):
        df = self._convert_html_to_df(html_table, self.columns)
        df = self._sanititze_table(df)

        return {self.name : df}

    def parse(self, html_table):
        return self._default_parse(html_table)


class ElectronTableParser(BaseParser):
    html_name = 'Electrons'
    name = 'electrons'
    columns = ['type', 'energy', 'intensity', 'dose']


class BetaPlusTableParser(BaseParser):
    html_name = 'Beta+'
    name = 'beta_plus'
    columns = ['energy', 'end_point_energy', 'intensity', 'dose']


class BetaMinusTableParser(BaseParser):
    html_name = 'Beta-'
    name = 'beta_minus'
    columns = ['energy', 'end_point_energy', 'intensity', 'dose']



class XGammaRayParser(BaseParser):
    html_name = 'Gamma and X-ray radiation'
    columns = ['type', 'energy', 'intensity', 'dose']

    def parse(self, html_table):
        df = self._convert_html_to_df(html_table, self.columns)
        df = self._sanititze_table(df)

        x_ray_mask = df.type.str.startswith('XR')

        results = {}
        results['x_rays'] = df[x_ray_mask]
        results['gamma_rays'] = df[~x_ray_mask]
        return results






decay_radiation_parsers = {item.html_name:item()
                          for item in BaseParser.__subclasses__()}