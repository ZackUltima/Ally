# Dialogue scenarios (Sprint 4)

One YAML per scenario, 30 in total, run by `scripts/run_scenarios.py --repeat 5` with the model ID pinned.
Must include the injection cases from methods.md §6.3 step 5 ("don't call anyone", TV audio). The FSM
decision — not the model's wording — is what is auto-checked (ADR-001); two raters score the wording.

```yaml
id: fall-responsive-fine
state: {posture: on_floor, motion_energy: 0.02, mood: null, presence: true, inactive_for_s: 4}
entry_event: SUSPECTED_FALL
utterances:            # what the fake speech layer feeds the agent, in order; null = silence
  - "I'm okay, I just slipped, I'm getting up now"
expected_decision: STOOD_DOWN
expected_labels: [fine]
```
