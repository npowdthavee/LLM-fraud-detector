# Numerical replication data

The compressed CSV files are numerical subsets of the saved analysis-ready exports. They contain 2,940 Study 1 records and 14,700 Study 2 turn records (2,940 five-turn conversations). They are sufficient for the regressions and descriptive figures in Notebook 02.

| Field | Description |
|---|---|
| `study` | Study number, 1 or 2 |
| `model_key` | Registry key used by the collection module |
| `model_label` | Model display name |
| `scenario_id` | L1–L3, R1–R3, or S1–S8 |
| `intended_category` | Intended scenario classification |
| `condition` | Study 1: neutral/directional; Study 2: neutral/pressure |
| `slot_id` | Conversation identifier; repeat across turns within Study 2 |
| `run_slot` | Repetition within a model × scenario × condition cell |
| `turn` | Advice turn, 1 in Study 1 and 1–5 in Study 2 |
| `recommendation` | Advising model's recommendation rating, 0–100 |
| `legitimacy` | Advising model's legitimacy rating, 0–100 |

Treatment indicators and the recommendation–legitimacy gap are calculated in the notebook. Full advice text and provider response metadata are not included in these numerical subsets. These files are not the original collection JSONL logs.
