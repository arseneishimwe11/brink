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


@app.command("run")
def run(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Phone video of the face or pile."),
    out: Path = typer.Option(None, help="Output folder. Default: data/derived/runs/<video name>/out"),
    material: str = typer.Option("unknown", help="Ground material; sets the slope limit. See the rules file."),
    site: str = typer.Option("unknown", help="Site name or id for the record."),
    gravity: str = typer.Option("auto", help="auto | sensor | scene"),
    sensors: Path = typer.Option(None, help="Sensor sidecar file. Default: <video>.sensors.json if present."),
    rules: Path = typer.Option(None, help="Rules YAML. Default: the built-in provisional rules."),
) -> None:
    """Check one video: 3D model, hazard map, JSON summary and PDF record."""
    from brink.domain import Site
    from brink.outputs import write_outputs
    from brink.pipeline import PipelineConfig, run_pipeline
    from brink.rules import load_rules

    work = DERIVED / "runs" / video.stem
    result = run_pipeline(
        video, work, PipelineConfig(material=material, gravity=gravity), load_rules(rules),
        Site(id=site, name=site, material=material), sensors,
    )
    paths = write_outputs(result, out or work / "out")
    recon = result.report.reconstruction
    if not recon.success:
        typer.echo(f"No result: {recon.failure_reason}")
    else:
        typer.echo(f"{len(result.report.hazards)} hazard(s) measured ({recon.frames_registered} frames, {sum(recon.timings_s.values()):.0f} s):")
        for hazard in result.report.hazards:
            typer.echo(f"  [{hazard.severity.value}, confidence {hazard.confidence:.2f}] {hazard.message}")
        if not result.report.hazards:
            typer.echo("  None measured. That is not a clearance.")
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")


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


@bench_app.command("full")
def bench_full(
    split: str = typer.Option("dev", help="dev (for tuning) or test (frozen)."),
    count: int = typer.Option(20, help="Scenes with geometric hazards and controls."),
    image_count: int = typer.Option(0, help="Extra scenes with crest cracks and puddles."),
    variants: str = typer.Option("sensor,scene", help="Pipeline variants to compare."),
    workers: int = typer.Option(2),
    log: str = typer.Option("", help="If set, append an entry with this title to docs/EXPERIMENTS.md."),
    notes: str = typer.Option("", help="Free text added to the experiment entry."),
) -> None:
    """Whole pipeline on rendered phone videos."""
    from brink.bench import report
    from brink.bench.full_tier import run_full_tier
    from brink.bench.suites import full_cases

    cases = full_cases(split, count, image_count)
    records = run_full_tier(cases, variants.split(","), DERIVED, workers)
    geometric = [r for r in records if not r["kind"].endswith(("crack", "puddle"))]
    summaries = report.group_summaries(geometric, ("variant",))
    body = (
        f"{len(cases)} rendered handheld videos ({count} with geometric hazards or controls, {image_count} with cracks or puddles).\n"
        "Each scene is tipped 0 to 4 degrees off true gravity. Variants: `sensor` reads gravity from the simulated\n"
        "phone sensor file, `scene` guesses it from the footage; others swap one stage.\n\n"
        "Reconstruction:\n\n" + report.reconstruction_table(records, "variant", "Variant")
        + "\n\nUndercuts and overhangs:\n\n" + report.hazard_table(summaries, ("Variant",))
        + "\n\nSlope and height of the test face:\n\n" + report.slope_table(summaries, ("Variant",))
    )
    command = f"brink bench full --split {split} --count {count} --image-count {image_count} --variants {variants}"
    report.save_section(DERIVED / "bench", f"full-{split}", f"Full tier (video in, hazards out), {split} split", command, body, records)
    report.write_results(DERIVED / "bench", RESULTS, SECTION_ORDER)
    if log:
        report.log_experiment(EXPERIMENTS, log, command, body, notes)
    typer.echo(body)


if __name__ == "__main__":
    app()
