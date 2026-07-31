import json
from pathlib import Path

import pandas as pd
import roman

from carsus.util import convert_atomic_number2symbol

OUTPUT_DIR = Path(__file__).resolve().parent
JOURNAL_CONFIG_DIR = OUTPUT_DIR / "journal_formats"


def _ion_stage_range(ion_numbers):
    """Format only present stages, compressing consecutive stages into runs."""
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


def _table_rows(summary, bold_total=True):
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


def _render_table(summary, config_path):
    """Render ``summary`` using a journal configuration file."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = "\n".join(
        _table_rows(summary, bold_total=config["bold_total"])
    )
    return config["template"].replace("{{TABLE_ROWS}}", rows)


def _build_summary(input_path):
    """Build an atomic-data summary from a Carsus HDF file."""
    with pd.HDFStore(Path(input_path), mode="r") as store:
        levels = store["levels_data"].reset_index()
        lines = store["lines_data"].reset_index()

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
    eligible_ions = counts_by_ion[
        (counts_by_ion["Levels"] > 1) & counts_by_ion["Lines"].notna()
    ].reset_index()

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
    summary.loc[len(summary)] = [
        "Total",
        "",
        summary["Levels"].sum(),
        summary["Lines"].sum(),
    ]
    return summary


JOURNAL_CONFIGS = {
    journal: JOURNAL_CONFIG_DIR / f"{journal}.json"
    for journal in ("aas", "aa", "mnras", "nature")
}


def exporttable(input_path, journal, output_stem=None):
    """Export an atomic-data summary table formatted for ``journal``.

    Parameters
    ----------
    input_path : path-like
        Carsus HDF file from which to build the summary.
    journal : {"aas", "aa", "mnras", "nature"}
        Journal whose LaTeX table format should be used.
    output_stem : str, optional
        Output filename without an extension. By default, files are named
        ``atomdata_summary_table_<journal>.tex`` and ``.txt``.
    """
    journal = journal.lower()
    try:
        config_path = JOURNAL_CONFIGS[journal]
    except KeyError as exc:
        supported = ", ".join(JOURNAL_CONFIGS)
        raise ValueError(
            f"Unsupported journal {journal!r}; choose one of: {supported}"
        ) from exc

    if output_stem is None:
        output_stem = f"atomdata_summary_table_{journal}"

    summary = _build_summary(input_path)
    (OUTPUT_DIR / f"{output_stem}.tex").write_text(
        _render_table(summary, config_path), encoding="utf-8"
    )
    (OUTPUT_DIR / f"{output_stem}.txt").write_text(
        summary.to_string(index=False), encoding="utf-8"
    )
    return summary
