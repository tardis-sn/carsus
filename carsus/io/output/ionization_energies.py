import copy

class IonizationEnergiesPreparer:
    def __init__(self, cmfgen_reader, ionization_energies):
        self.ionization_energies = ionization_energies

        if (cmfgen_reader is not None) and hasattr(
            cmfgen_reader, "ionization_energies"
        ):
            combined_ionization_energies = copy.deepcopy(ionization_energies)
            combined_ionization_energies.base = (
                cmfgen_reader.ionization_energies.combine_first(
                    ionization_energies.base
                )
            )
            self.ionization_energies = combined_ionization_energies

    @property
    def ionization_energies_prepared(self):
        """
        Prepare the DataFrame with ionization energies for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """
        ionization_energies_prepared = self.ionization_energies.base.copy()
        ionization_energies_prepared = ionization_energies_prepared.reset_index()
        ionization_energies_prepared["ion_charge"] += 1
        ionization_energies_prepared = ionization_energies_prepared.rename(
            columns={"ion_charge": "ion_number"}
        )
        ionization_energies_prepared = ionization_energies_prepared.set_index(
            ["atomic_number", "ion_number"]
        )

        return ionization_energies_prepared.squeeze()