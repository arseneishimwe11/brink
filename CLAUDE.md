# Brink: standing instructions

## What this is
Brink turns handheld phone videos of pits, quarries and excavations into 3D
models and flags dangerous ground shapes. Full context: docs/PROJECT_BRIEF.md.
Read it before planning any work.

## Ownership
This codebase is yours: you own and maintain it long-term, and its design is
your call. Build foundations meant to last (clear modules, typed interfaces,
tests), not demo code.

## Environment
- We work in Claude Code on the web, in a cloud sandbox. First check the
  machine (CPU, RAM, GPU, disk) and size everything to fit. Assume no GPU.
- Network access may be limited to package registries and GitHub. If a download
  is blocked, choose another source or tool, and tell me what was blocked.
- Some popular tools need an NVIDIA GPU for certain steps (for example COLMAP's
  dense stereo). Choose CPU-capable options, and keep a clean switch for GPU later.
- Use a project virtual environment.
- Large videos live in data/raw/ (Git LFS if needed). Never commit derived large
  outputs; regenerate them from scripts.

## Licensing (this will be a commercial product)
- Record the license of every dependency and model in docs/LICENSES.md.
- Prefer permissive licenses (MIT, BSD, Apache-2.0).
- Flag AGPL, GPL and non-commercial licenses before relying on them. Several
  well-known 3D and AI reconstruction models are non-commercial only.

## Honesty rules
- Every accuracy claim comes from a measured run, with the command that produced it.
- Never tune on held-out test data. If a result does not reproduce, say so.
- No output ever states that ground is "safe". Outputs report measured hazards,
  measurements and confidence.

## Working style
- Keep going until the phase's work is complete. Diagnose ordinary obstacles
  (failed installs, slow steps, errors) and work through them instead of stopping.
- Ask only for genuine decisions, or for data only I can provide (phone videos).
- No approval needed for: creating and editing files in this repo, installing
  packages into the venv, downloading open-source tools and models, running long
  jobs, commits.
- Ask before: deleting or changing anything in data/raw/, spending money, or
  touching anything outside this repo.
- Every few steps, give a one-line update. End each session with a plain-language
  report that a reader with no context can follow: what works, measured numbers,
  what failed, and what comes next.

## Commit messages
- Write like a human developer: a short, plain summary of what changed and why.
- Subject line: imperative mood, under 60 characters, no period.
  Examples: "Add undercut detector with profile slicing",
  "Fix scale estimate when tape marks are missing".
- Add a body only when the reason isn't obvious: 1 to 3 short lines.
- One commit per logical change. Summarize; don't list every file.
- No em dashes, no double hyphens, no emojis, no bullet lists in the subject.
- No trailers or footers: no Co-Authored-By, no "Generated with", no session links.
