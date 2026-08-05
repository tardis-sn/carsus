"""Build journal-formatted atomic data summary tables."""

import json
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
import roman

from carsus.util import (
    convert_atomic_number2symbol,
    convert_symbol2atomic_number,
)

OUTPUT_DIR = Path(__file__).resolve().parent
JOURNAL_CONFIG_DIR = OUTPUT_DIR / "journal_formats"


def _ion_stage_range(ion_numbers: Iterable[int]) -> str:
    """Format present ion stages as consecutive ranges.

    Parameters
    ----------
    ion_numbers : typing.Iterable[int]
        Zero-based ion numbers to format.

    Returns
    -------
    str
        Ion stages in spectroscopic notation.
    """
    # finding consecutive ion stage ranges
    stages = sorted({int(ion) + 1 for ion in ion_numbers})
    runs = []
    start = previous = stages[0]

    for stage in stages[1:]:
        if stage != previous + 1:
            runs.append((start, previous))
            start = stage
        previous = stage
    runs.append((start, previous))

    return ", ".join(
        roman.toRoman(start)
        if start == end
        else f"{roman.toRoman(start)}--{roman.toRoman(end)}"
        for start, end in runs
    )


def _table_rows(
    summary: pd.DataFrame, bold_total: bool = True
) -> List[str]:
    """Convert a summary table into LaTeX rows.

    Parameters
    ----------
    summary : pandas.DataFrame
        Atomic data summary to convert.
    bold_total : bool, optional
        Whether to make the total row bold.

    Returns
    -------
    typing.List[str]
        Formatted LaTeX table rows.
    """
    # converting summary columns into latex rows
    elements = summary["Element"].astype(str)
    stages = summary["Ion stages"].astype(str)
    levels = summary["Levels"].astype(str)
    lines = summary["Lines"].astype(str)

    rows = elements.str.cat([stages, levels, lines], sep=" & ") + r" \\"

    if bold_total:
        bold_rows = (
            r"\textbf{" + elements + "} & " + stages + r" & \textbf{"
            + levels + r"} & \textbf{" + lines + r"} \\"
        )
        rows = rows.mask(elements.eq("Total"), bold_rows)

    return rows.tolist()


def _render_table(summary: pd.DataFrame, config_path: Path) -> str:
    """Render `summary` using a journal configuration file.

    Parameters
    ----------
    summary : pandas.DataFrame
        Atomic data summary to render.
    config_path : pathlib.Path
        Path to the journal configuration file.

    Returns
    -------
    str
        Complete LaTeX document containing the summary table.
    """
    # loading the selected journal template
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = "\n".join(
        _table_rows(summary, bold_total=config["bold_total"])
    )
    return config["template"].replace("{{TABLE_ROWS}}", rows)


def _build_summary(
    input_path: Union[str, Path], elements: Optional[List[str]] = None
) -> pd.DataFrame:
    """Build an atomic data summary from a Carsus HDF file.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the Carsus HDF file.
    elements : typing.List[str], optional
        Chemical symbols to include. All available elements are included by
        default.

    Returns
    -------
    pandas.DataFrame
        Atomic data counts and ion stages grouped by element.
    """
    # loading level and line data
    with pd.HDFStore(Path(input_path), mode="r") as store:
        levels = store["levels_data"].reset_index()
        lines = store["lines_data"].reset_index()

    # combining level and line counts by ion
    counts_by_ion = pd.concat(
        [
            levels.groupby(["atomic_number", "ion_number"])
            .size()
            .rename("Levels"),
            lines.groupby(["atomic_number", "ion_number"])
            .size()
            .rename("Lines"),
        ],
        axis=1,
    )
    # keeping ions with usable level and line data
    eligible_ions = counts_by_ion[
        (counts_by_ion["Levels"] > 1) & counts_by_ion["Lines"].notna()
    ].reset_index()

    if elements is not None:
        # filtering to the requested elements
        atomic_numbers = [
            convert_symbol2atomic_number(element) for element in elements
        ]
        eligible_ions = eligible_ions[
            eligible_ions["atomic_number"].isin(atomic_numbers)
        ]

    # summarizing ion stages and counts by element
    ion_stages = (
        eligible_ions
        .groupby("atomic_number")["ion_number"]
        .apply(_ion_stage_range)
        .rename("Ion stages")
    )
    counts_by_element = eligible_ions.groupby("atomic_number")[
        ["Levels", "Lines"]
    ].sum()
    summary = pd.concat(
        [
            ion_stages,
            counts_by_element,
        ],
        axis=1,
    )
    # formatting element names for latex
    summary.insert(
        0,
        "Element",
        [
            rf"$\mathrm{{{convert_atomic_number2symbol(z)}}}_{{{z}}}$"
            for z in summary.index
        ],
    )
    summary = summary.reset_index(drop=True)
    summary[["Levels", "Lines"]] = summary[["Levels", "Lines"]].astype(int)
    # adding the table total
    summary.loc[len(summary)] = [
        "Total",
        "",
        summary["Levels"].sum(),
        summary["Lines"].sum(),
    ]
    return summary


JOURNAL_CONFIGS = {
    journal: JOURNAL_CONFIG_DIR / f"{journal}.json"
    for journal in ("aas", "aa", "mnras", "nature", "custom")
}


def exporttable(
    input_path: Union[str, Path],
    journal: str,
    output_filename: Optional[str] = None,
    elements: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Export an atomic data summary table formatted for `journal`.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the Carsus HDF file from which to build the summary.
    journal : {"aas", "aa", "mnras", "nature", "custom"}
        Journal whose LaTeX table format should be used.
        The custom option reads ``journal_formats/custom.json``.
    output_filename : str, optional
        Output filename without an extension. By default, files are named
        ``atomdata_summary_table_<journal>.tex`` and ``.txt``.
    elements : list of str, optional
        Chemical symbols to include, such as ``["H", "Si", "Fe"]``.
        By default, all available elements are included.

    Returns
    -------
    pandas.DataFrame
        Atomic data summary written to the output files.
    """
    # selecting the journal configuration
    journal = journal.lower()
    try:
        config_path = JOURNAL_CONFIGS[journal]
    except KeyError as exc:
        supported = ", ".join(JOURNAL_CONFIGS)
        raise ValueError(
            f"Unsupported journal {journal!r}; choose one of: {supported}"
        ) from exc

    if output_filename is None:
        output_filename = f"atomdata_summary_table_{journal}"

    # building and writing the output tables
    summary = _build_summary(input_path, elements=elements)
    (OUTPUT_DIR / f"{output_filename}.tex").write_text(
        _render_table(summary, config_path), encoding="utf-8"
    )
    (OUTPUT_DIR / f"{output_filename}.txt").write_text(
        summary.to_string(index=False), encoding="utf-8"
    )
    return summary
