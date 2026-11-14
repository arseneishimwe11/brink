"""Brink command line."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Brink: hazard checks from one phone video.")
bench_app = typer.Typer(no_args_is_help=True, help="Run benchmarks and write docs/RESULTS.md.")
app.add_typer(bench_app, name="bench")

DERIVED = Path("data/derived")
RESULTS = Path("docs/RESULTS.md")
EXPERIMENTS = Path("docs/EXPERIMENTS.md")
SECTION_ORDER = ["full-test", "full-dev", "known-pose-test", "known-pose-dev", "geometry-test", "geometry-dev"]


@bench_app.command("geometry")
def bench_geometry(
    split: str = typer.Option("dev", help="dev (for tuning) or test (frozen)."),
    count: int = typer.Option(40, help="Number of scenes."),
    methods: str = typer.Option("profile,normals", help="Undercut methods to compare."),
    conditions: str = typer.Option("ideal,typical,poor", help="Simulated reconstruction quality."),
    workers: int = typer.Option(4),
    log: str = typer.Option("", help="If set, append an entry with this title to docs/EXPERIMENTS.md."),
    notes: str = typer.Option("", help="Free text added to the experiment entry."),
) -> None:
    """Hazard geometry alone, on sampled ground-truth surfaces."""
    from brink.bench import report
    from brink.bench.geometry_tier import run_geometry_tier
    from brink.bench.suites import geometry_cases

    cases = geometry_cases(split, count)
    records = run_geometry_tier(cases, methods.split(","), conditions.split(","), DERIVED / "synth_cache", workers)
    summaries = report.group_summaries(records, ("method", "condition"))
    names = ("Undercut method", "Cloud quality")
    body = (
        f"{count} synthetic scenes, hazard geometry run on sampled surfaces (no images).\n"
        "Cloud quality sets depth noise, holes, and the assumed error in gravity and scale:\n"
        "ideal (0.15% of range, 0 deg, 0%), typical (0.4%, 1 deg, 5%), poor (0.8%, 3 deg, 12%).\n\n"
        "Undercuts and overhangs:\n\n" + report.hazard_table(summaries, names)
        + "\n\nSlope and height of the test face:\n\n" + report.slope_table(summaries, names)
    )
    command = f"brink bench geometry --split {split} --count {count} --methods {methods} --conditions {conditions}"
    name = f"geometry-{split}"
    report.save_section(DERIVED / "bench", name, f"Geometry tier, {split} split", command, body, records)
    report.write_results(DERIVED / "bench", RESULTS, SECTION_ORDER)
    if log:
        report.log_experiment(EXPERIMENTS, log, command, body, notes)
    typer.echo(body)


if __name__ == "__main__":
    app()
