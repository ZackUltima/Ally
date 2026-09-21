# docs/

`Ally_Project_Plan.html` (repo root, v1.1) is the graded, human-readable master plan. The files in
`context/` are that plan split into focused reference docs so a Claude Code session can be pointed at
exactly one of them per task. When the plan and a context doc disagree, fix both and note it in
`context/decisions.md`.

| Task touches… | Point Claude at |
|---|---|
| pipeline, queues, FSM states, privacy boundary, module contracts | `context/architecture.md` |
| user stories, FR/NFR, latency and false-alarm budgets | `context/requirements.md` |
| fall / FER / VLM / dialogue methods, datasets, training and eval steps | `context/methods.md` |
| tests, replay harness, scenario harness, result tables | `context/evaluation.md` |
| consent, retention, cloud boundary, security, limitations | `context/ethics.md` |
| what is due when, sprint exit criteria | `context/schedule.md` |
| why something is the way it is; before changing any number | `context/decisions.md` |

`journal.md` — five lines every Friday (done / measured / blocked / decided / next). It becomes the
implementation chapter of the final report.

`proposal/`, `srd/`, `diagrams/`, `final_report/` — course deliverables (created as they are written).
