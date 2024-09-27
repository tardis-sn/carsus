import hashlib
import os
import logging
import re
import shutil
from functools import lru_cache

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.utils.data import download_file

logger = logging.getLogger(__name__)

class HDFWriterMixin(object):
    def __new__(cls, *args, **kwargs):
        instance = super(HDFWriterMixin, cls).__new__(cls)
        instance.optional_hdf_properties = []
        instance.__init__(*args, **kwargs)
        return instance

    @staticmethod
    def to_hdf_util(
        path_or_buf, path, elements, overwrite, complevel=9, complib="blosc"
    ):
        """
        A function to uniformly store TARDIS data to an HDF file.

        Scalars will be stored in a Series under path/scalars
        1D arrays will be stored under path/property_name as distinct Series
        2D arrays will be stored under path/property_name as distinct DataFrames

        Units will be stored as their CGS value

        Parameters
        ----------
        path_or_buf : str or pandas.io.pytables.HDFStore
            Path or buffer to the HDF file
        path : str
            Path inside the HDF file to store the `elements`
        elements : dict
            A dict of property names and their values to be
            stored.
        overwrite : bool
            If the HDF file path already exists, whether to overwrite it or not

        Notes
        -----
        `overwrite` option doesn't have any effect when `path_or_buf` is an
        HDFStore because the user decides on the mode in which they have
        opened the HDFStore ('r', 'w' or 'a').
        """
        if (
            isinstance(path_or_buf, str)
            and os.path.exists(path_or_buf)
            and not overwrite
        ):
            raise FileExistsError(
                "The specified HDF file already exists. If you still want "
                "to overwrite it, set function parameter overwrite=True"
            )

        else:
            try:  # when path_or_buf is a str, the HDFStore should get created
                buf = pd.HDFStore(
                    path_or_buf, complevel=complevel, complib=complib
                )
            except TypeError as e:
                if str(e) == "Expected bytes, got HDFStore":
                    # when path_or_buf is an HDFStore buffer instead
                    logger.debug(
                        "Expected bytes, got HDFStore. Changing path to HDF buffer"
                    )
                    buf = path_or_buf
                else:
                    raise e

        if not buf.is_open:
            buf.open()

        scalars = {}
        for key, value in elements.items():
            if value is None:
                value = "none"
            if hasattr(value, "cgs"):
                value = value.cgs.value
            if np.isscalar(value):
                scalars[key] = value
            elif hasattr(value, "shape"):
                if value.ndim == 1:
                    # This try,except block is only for model.plasma.levels
                    try:
                        pd.Series(value).to_hdf(buf, os.path.join(path, key))
                    except NotImplementedError:
                        logger.debug(
                            "Could not convert SERIES to HDF. Converting DATAFRAME to HDF"
                        )
                        pd.DataFrame(value).to_hdf(buf, os.path.join(path, key))
                else:
                    pd.DataFrame(value).to_hdf(buf, os.path.join(path, key))
            else:  # value is a TARDIS object like model, transport or plasma
                try:
                    value.to_hdf(buf, path, name=key, overwrite=overwrite)
                except AttributeError:
                    logger.debug(
                        "Could not convert VALUE to HDF. Converting DATA (Dataframe) to HDF"
                    )
                    data = pd.DataFrame([value])
                    data.to_hdf(buf, os.path.join(path, key))

        if scalars:
            pd.Series(scalars).to_hdf(buf, os.path.join(path, "scalars"))

        if buf.is_open:
            buf.close()

    def get_properties(self):
        data = {name: getattr(self, name) for name in self.full_hdf_properties}
        return data

    @property
    def full_hdf_properties(self):
        if hasattr(self, "virt_logging") and self.virt_logging:
            self.hdf_properties.extend(self.vpacket_hdf_properties)

        return self.optional_hdf_properties + self.hdf_properties

    @staticmethod
    def convert_to_snake_case(s):
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def to_hdf(self, file_path_or_buf, path="", name=None, overwrite=False):
        """
        Parameters
        ----------
        file_path_or_buf : str or pandas.io.pytables.HDFStore
            Path or buffer to the HDF file
        path : str
            Path inside the HDF file to store the `elements`
        name : str
            Group inside the HDF file to which the `elements` need to be saved
        overwrite : bool
            If the HDF file path already exists, whether to overwrite it or not
        """
        if name is None:
            try:
                name = self.hdf_name
            except AttributeError:
                name = self.convert_to_snake_case(self.__class__.__name__)
                logger.debug(
                    f"self.hdf_name not present, setting name to {name} for HDF"
                )

        data = self.get_properties()
        buff_path = os.path.join(path, name)
        self.to_hdf_util(file_path_or_buf, buff_path, data, overwrite)


class PlasmaWriterMixin(HDFWriterMixin):
    def get_properties(self):
        data = {}
        if self.collection:
            properties = [
                name
                for name in self.plasma_properties
                if isinstance(name, tuple(self.collection))
            ]
        else:
            properties = self.plasma_properties
        for prop in properties:
            for output in prop.outputs:
                data[output] = getattr(prop, output)
        data["atom_data_uuid"] = self.atomic_data.uuid1
        if "atomic_data" in data:
            data.pop("atomic_data")
        if "nlte_data" in data:
            logger.warning("nlte_data can't be saved")
            data.pop("nlte_data")
        return data

    def to_hdf(
        self,
        file_path_or_buf,
        path="",
        name=None,
        collection=None,
        overwrite=False,
    ):
        """
        Parameters
        ----------
        file_path_or_buf : str or pandas.io.pytables.HDFStore
            Path or buffer to the HDF file
        path : str
            Path inside the HDF file to store the `elements`
        name : str
            Group inside the HDF file to which the `elements` need to be saved
        collection :
            `None` or a `PlasmaPropertyCollection` of which members are
            the property types which will be stored. If `None` then
            all types of properties will be stored. This acts like a filter,
            for example if a value of `property_collections.basic_inputs` is
            given, only those input parameters will be stored to the HDF file.
        overwrite : bool
            If the HDF file path already exists, whether to overwrite it or not
        """
        self.collection = collection
        super(PlasmaWriterMixin, self).to_hdf(
            file_path_or_buf, path, name, overwrite
        )


@lru_cache(maxsize=None)
def download_from_url(url, dst, checksum, src=None, retries=3):
    """Download files from a given URL

    Parameters
    ----------
    url : str
        URL to download from
    dst : str
        Destination folder for the downloaded file
    src : tuple
        List of URLs to use as mirrors
    """

    cached_file_path = download_file(url, sources=src, pkgname="tardis")

    with open(cached_file_path, "rb") as f:
        new_checksum = hashlib.md5(f.read()).hexdigest()

    if checksum == new_checksum:
        shutil.copy(cached_file_path, dst)

    elif checksum != new_checksum and retries > 0:
        retries -= 1
        logger.warning(
            f"Incorrect checksum, retrying... ({retries+1} attempts remaining)"
        )
        download_from_url(url, dst, checksum, src, retries)

    else:
        logger.error("Maximum number of retries reached. Aborting")
