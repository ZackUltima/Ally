# Ally

A laptop + webcam care companion that detects falls, prolonged inactivity and low mood, **talks to the person
first**, and escalates to a guardian on Telegram only after verification. Raw video never leaves the device.

CAT405 final-year project, Universiti Sains Malaysia, 2026/27. **Research prototype — not a medical or
emergency device.** The graded plan is `Ally_Project_Plan.html`; working notes live in `docs/context/`.

## Setup (Windows 11, RTX 2050, Python 3.11)

Python 3.11 is required (3.13 is installed system-wide but CV/ML wheels lag — do not use it). The venv is
managed with [uv](https://docs.astral.sh/uv/), which installs 3.11 without admin rights.

```powershell
# one-time: install uv (user scope)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# in the repo
.\tasks.ps1 venv          # uv python install 3.11 + uv venv .venv
.\tasks.ps1 install-dev   # pydantic, numpy, ruff, pytest, pre-commit  (enough for make test / lint)
.\tasks.ps1 install       # + CUDA torch, ultralytics, opencv, speech, fastapi …  (several GB)
.\tasks.ps1 lock          # freeze exact pins into requirements*.txt with --universal (CI is Linux); commit them.
                          # requirements-dev.txt is committed; requirements.txt appears after the first full resolve
copy .env.example .env    # then fill in keys — .env is git-ignored
```

`tasks.ps1` mirrors the `Makefile` target for target; use `make …` from Git Bash if GNU make is installed.
Before the first `install`, confirm the CUDA wheel tag (`TORCH_INDEX`, default `cu126`) against
<https://pytorch.org/get-started/locally/> for the installed driver.

## Everyday commands

| Command | What |
|---|---|
| `make test` / `.\tasks.ps1 test` | pytest `tests/unit` + `tests/replay` (CPU only, no keys) |
| `make lint` / `fmt` | ruff check + format |
| `make replay SESSION=<name>` | recorded session → perception → fusion → agent, prints events |
| `make scenarios` | 30 scripted dialogue scenarios (needs `ANTHROPIC_API_KEY`) |
| `make run` | live app · `make bench` FPS/VRAM of the pose model on the webcam |
| `make train-fall` / `eval-fall` / `train-fer` / `eval-fer` | experiments; fixed seeds; write to `results/` |

## Layout

```
ally/        perception · fusion (pure FSM) · agent · speech · notify · api · storage · contracts.py · config.py · main.py
scripts/     benchmark_fps · extract_keypoints · record_session · prepare_urfall · train_/eval_ fall & fer · run_scenarios
tests/       unit/ · replay/<session>/expected_events.json · scenarios/*.yaml
docs/        context/ (architecture, requirements, methods, evaluation, ethics, schedule, decisions) · journal.md
data/ models/ results/   git-ignored
```

Rules every change must respect are in `CLAUDE.md` (the FSM owns escalation; no raw frames on disk; only
text + one keyframe per event leaves the device; numbers change only with an ADR).

## Licence and data

Code: to be decided before the SRD. UR Fall Detection Dataset is CC BY-NC-SA 4.0; FER-2013 under its Kaggle
terms. No dataset or recording is committed to this repository.
