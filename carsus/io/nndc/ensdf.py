class ENSDFReader:

    def __init__(self, fname):

        #### Still seemingly missing several bits of data from NNDC ###
        raise NotImplementedError('Not quite working yet')
        raw_decay_list = ensdf.decays(fname)[0]

        self.parent_nuc_id = raw_decay_list[0]
        self.daughter_nuc_id = raw_decay_list[1]
        self.reaction_id = raw_decay_list[2]
        self.half_life = raw_decay_list[3]  # in s
        self.half_life_uncert = raw_decay_list[4]  # in s
        self.branching_ratio = raw_decay_list[5]  # percent
        self.conversion_factor_gamma = raw_decay_list[6]
        self.conversion_factor_gamma_uncert = raw_decay_list[7]
        # for gamma intensity to photons per 100 decays of the parent

        self.conversion_factor_lepton = raw_decay_list[8]
        self.conversion_factor_lepton_uncert = raw_decay_list[9]
        # Conversion factor for electron capture / beta intensity to
        # electron captures / betas per 100 decays of the parent

        self.gamma = self._gamma_to_dataframe(raw_decay_list[11])
        self.alpha = self._alpha_to_dataframe(raw_decay_list[12])
        self.beta_minus = self._beta_minus_to_dataframe(raw_decay_list[13])
        self.beta_plus = self._beta_minus_to_dataframe(raw_decay_list[14])

    @staticmethod
    def _gamma_to_dataframe(gamma_list):
        columns = ['level_1_id', 'level_2_id', 'parent_id', 'daughter_id',
                   'energy', 'energy_uncert', 'intensity', 'intensity_uncert',
                   'electron_conversion_intensity', 'electron_conversion_intensity_uncert',
                   'total_transition_intensity', 'total_transition_intensity_uncert']
        columns += ['{0}_electron_conversion_intensity'.format(item) for item in 'klm']

        return pd.DataFrame(data=gamma_list, columns=columns)

    @staticmethod
    def _alpha_to_dataframe(alpha_list):
        if alpha_list == []:
            return None
        else:
            raise NotImplementedError('Not implemented yet')

        # parent nuclide id in state_id form
        # child nuclide id in state_id form
        # alpha energy
        # alpha intensity in percent of total alphas

    @staticmethod
    def _beta_minus_to_dataframe(beta_minus_list):
        if beta_minus_list == []:
            return None
        columns = ['parent_id', 'child_id', 'endpoint_energy', 'average_energy', 'intensity']
        return pd.DataFrame(beta_minus_list, columns=columns)

    @staticmethod
    def _beta_plus_to_dataframe(beta_plus_list):
        if beta_plus_list == []:
            return None
        columns = ['parent_id', 'child_id', 'endpoint_energy', 'average_energy', 'intensity',
                   'electron_capture_intensity']
        columns += ['{0}_electron_conversion_intensity'.format(item) for item in 'klm']
        return pd.DataFrame(beta_plus_list, columns=columns)
