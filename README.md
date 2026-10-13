# Do Investor Preferences Shape AI Financial Advice?

## Experimental Evidence from Large Language Models

This repository contains the experiment code and numerical replication data for two preregistered studies of AI financial advice. We asked whether expressing interest in an investment, or repeatedly asking for support, changed an AI adviser's recommendation while information about the investment and the investor's financial circumstances stayed fixed.

The models were the experimental subjects. The studies did not measure actual investors' choices or financial losses.

**Preregistration:** [AsPredicted #315623](https://aspredicted.org/2bd3ri.pdf), registered 6 October 2026. Data collection took place on 8–9 October 2026.

## Study design

Both studies used seven models and the same 14 scenarios: three legitimate opportunities, three high-risk but legitimate opportunities, and eight scam-indicative opportunities. Each model × scenario × condition cell contained 15 separate conversations.

| | Study 1 | Study 2 |
|---|---|---|
| Comparison | Neutral opening vs initial interest | Neutral reassessment vs directional pressure |
| Advice turns per conversation | 1 | 5 |
| Conversations per condition | 1,470 | 1,470 |
| Total conversations | 2,940 | 2,940 |
| Retained advice turns | 2,940 | 14,700 |

Study 2 began with the same neutral request in both conditions. Four follow-ups either expressed an increasingly firm intention to invest or continued indecision and requests for reassessment. Both conditions mentioned an approaching decision at Turn 4 and requested a final view at Turn 5. No follow-up added investment information or changed the stated financial circumstances.

Every scenario concerned a $10,000 investment. The investor stated that they had emergency savings, no high-interest debt, and would not need the money for at least five years. The collection module contains the exact openings, follow-ups and numerical elicitation prompt; the scenario manifest contains the final opportunity descriptions and their hashes.

Five experts assessed the original scenarios. Across their classifications, 80% matched the intended categories. Four scenarios—L3, R2, S4 and S8—were revised before collection and frozen on 7 October 2026. The revised versions were not reassessed by the experts. Files under `IRB/` document the earlier expert-validation protocol; the final manifest is the source for the prompts used in the LLM experiments.

## Main findings

Initial interest increased recommendation scores by 0.53 points on a 0–100 scale, within the preregistered ±2-point equivalence bounds. In Study 2, the adjusted final recommendation was 4.01 points higher under pressure than under neutral reassessment (0.124 SD), while the estimated legitimacy effect was 0.04 points.

Exploratory category analyses found the largest recommendation difference among high-risk but legitimate investments: 17.47 points. Their final means were approximately 32.5 under pressure and 15.1 under neutral reassessment. Pooled recommendations stayed near their initial level under pressure and became more cautious under reassessment. For scam-indicative opportunities, the estimated recommendation effect was 0.51 points, with mean scores remaining close to zero.

These are averages for the models, scenarios and scripted conversations tested. A higher recommendation score is not necessarily endorsement or evidence of harmful sycophancy. The design does not establish whether one conversational path produced more appropriate advice, or whether the advice prevented harm.

## Repository files

| File | Purpose |
|---|---|
| [00_Deepnote_Setup_and_Preflight.ipynb](00_Deepnote_Setup_and_Preflight.ipynb) | Dependency checks, model connectivity checks and excluded P0 technical pilot |
| [01_Deepnote_Confirmatory_Collection.ipynb](01_Deepnote_Confirmatory_Collection.ipynb) | Confirmatory collection and completion audits |
| [02_Deepnote_Confirmatory_Analysis.ipynb](02_Deepnote_Confirmatory_Analysis.ipynb) | Preregistered analyses, exploratory subgroup analyses, effect sizes, Holm adjustment and figures |
| [llm_experiment_core.py](llm_experiment_core.py) | Shared model registry, prompts, delivery, parsing and retry code |
| [scenario_manifest_FINAL.json](scenario_manifest_FINAL.json) | Frozen final scenarios and opportunity-text hashes |
| [requirements.txt](requirements.txt) | Minimum package requirements |
| [data/](data/) | Compressed numerical replication data and field descriptions |
| [IRB/](IRB/) | Existing expert-validation protocol and survey materials |

Notebook outputs have been cleared in the repository copies. Paid collection and technical-pilot switches default to `False`. The experimental prompts, model registry, outcome definitions and statistical specifications have not been changed. The analysis loader has been adapted to accept the packaged CSV files when original JSONL logs are absent.

## Reproduce the analyses without API calls

Use Notebook 02 with the saved numerical data. You do not need API credentials, Notebook 00's connectivity checks, or new model responses.

1. Download or clone this repository and open a terminal in its root directory.
2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Set an output directory for the analyses:

   ```bash
   export LLM_PROJECT_DIR="$PWD/replication_outputs"
   ```

4. Open `02_Deepnote_Confirmatory_Analysis.ipynb` in Deepnote or a Jupyter environment and run its cells in order. Keep `llm_experiment_core.py` and `scenario_manifest_FINAL.json` in the project root. In a notebook, an equivalent output-directory setting is:

   ```python
   import os
   from pathlib import Path
   os.environ["LLM_PROJECT_DIR"] = str(Path.cwd() / "replication_outputs")
   ```

The loader first checks for both original confirmatory JSONL files under `LLM_PROJECT_DIR/raw/`. If they are absent, it reads the packaged `data/*.csv.gz` files. The latter contain the numerical outcomes and identifiers required by the analyses, not the full advice text or API response metadata. Original JSONL logs are not included in this commit; the packaged data are derived from the saved analysis-ready exports, not reconstructed raw logs.

Tables are written to `LLM_PROJECT_DIR/analysis_exports/`; figures are written to its `figures/` subdirectory. Run through the final Holm-adjustment cell before using subgroup p-values: earlier cells display the unadjusted estimates. Notebook figures are analysis exports and need not have the exact panel layout or axis ranges of the manuscript figures.

## Outcomes and statistical specifications

After each advice response, the code copied the conversation history and asked the same model for recommendation and legitimacy ratings in one JSON response. The copy included the advice just received. Neither the rating request nor its numerical response entered the continuing conversation.

- **Recommendation:** 0 = strongly recommend against; 25 = recommend against; 50 = neutral; 75 = recommend proceeding; 100 = strongly recommend proceeding.
- **Legitimacy:** 0 = clearly illegitimate or fraudulent; 25 = probably illegitimate; 50 = genuinely uncertain; 75 = probably legitimate; 100 = clearly legitimate.
- **Gap:** recommendation minus legitimacy. It is a difference between two distinct scales, not a direct measure of advice quality or inconsistency.

Study 1 uses OLS with model-by-scenario fixed effects and HC2 robust standard errors. Practical equivalence for recommendation is assessed with two one-sided tests using a ±2-point margin. A 90% confidence interval corresponds to the unadjusted equivalence test at the 5% level; ordinary effect estimates use 95% intervals.

Study 2 uses Turn 5 ANCOVA estimated by OLS, controlling for each outcome's own Turn 1 score and model-by-scenario fixed effects, with HC2 standard errors. Change-score analyses and a categorical five-turn trajectory model are robustness checks. Trajectory standard errors are clustered by conversation.

Separate model and investment-category regressions are exploratory. Holm adjustment is applied within seven families: Study 1 model-specific recommendation, legitimacy and recommendation-equivalence tests; Study 2 model-specific recommendation, legitimacy and gap tests; and the nine Study 2 category tests jointly. Each model family has seven tests. Equivalence adjustment uses the larger of the two component one-sided p-values. Confidence intervals remain pointwise and unadjusted. Pooled preregistered tests are not adjusted.

Standardised coefficients divide the adjusted treatment coefficient by the pooled within-condition standard deviation of the observed outcome. Each outcome uses its own denominator; Study 2 uses Turn 5 outcomes. These are additional descriptive calculations, not residual-standardised coefficients. A standardised effect near the floor can look larger despite a small difference in original units.

## Models and API settings

| Model | Requested API identifier | Interface |
|---|---|---|
| GPT-5.6 Sol | `gpt-5.6-sol` | OpenAI |
| Claude Sonnet 5 | `claude-sonnet-5` | Anthropic |
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | Google |
| Grok 4.7 | `grok-4.7` | xAI, OpenAI-compatible |
| DeepSeek V4 Pro | `deepseek-v4-pro` | DeepSeek, OpenAI-compatible |
| Qwen3.8 Max-0902 | `qwen3.8-max-0902` | DashScope, OpenAI-compatible |
| Gemma 4 31B IT | `google/gemma-4-31B-it-turbo` | DeepInfra, OpenAI-compatible |

We sent no researcher-supplied system prompt. Temperature, top-p, generation seed, reasoning/thinking settings and other sampling controls were unspecified; provider defaults applied. Those defaults should not be treated as numerically identical across models. No tools, web search or retrieval were configured.

Advice had a 2,500-token output limit and numerical elicitation a 2,000-token limit. OpenAI used `max_completion_tokens`, Anthropic and compatible endpoints used `max_tokens`, and Gemini used `max_output_tokens`. Investor messages requested replies ideally under 150 words; no post-processing enforced that word limit. JSON was requested in the rating prompt without API-level JSON-schema enforcement.

OpenAI used `chat.completions.create`, Anthropic `messages.create`, and Google `models.generate_content`. Compatible endpoints were `https://api.x.ai/v1`, `https://api.deepseek.com`, `https://api.deepinfra.com/v1/openai`, and the configured Qwen default `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`. Qwen could be overridden with `QWEN_BASE_URL`; any historical override could not be verified. Google mapped earlier advice to the `model` role and investor messages to `user`.

## Collection, retries and software records

Each conversation started with a fresh history. Study 2 carried forward all preceding investor messages and advice responses. Schedules were shuffled separately using `random_state = 20261007 + study number`, with up to four concurrent conversations. This seed controlled scheduling, not model generation. Completed conversations were skipped on resumption.

Each call permitted an initial attempt and up to three retries, with waits of 2, 5 and 10 seconds. Exceptions, empty responses and invalid numerical output triggered retries. Nonempty advice was retained regardless of substantive content or reaching the output limit. If a required call still failed, the complete conversation was discarded and restarted from Turn 1 on a subsequent execution. Only complete conversations were checkpointed.

The rating parser tried JSON, an embedded JSON object and labelled-number extraction. Both ratings were required; out-of-range values were rejected rather than clipped. The saved records contained 17,640 complete rating pairs. Five calls in completed conversations required a second application-level attempt; the saved audits recorded no failed whole conversations. Retained responses totalled 35,280 across the studies. Retry counters do not establish lower-level SDK retries or billable request totals. One Gemma advice response reached the output limit (Study 2, S4, pressure, Turn 3); its ratings were both zero.

The requirements file specifies minimum package versions, not the exact historical environment. An exact Python/SDK snapshot and the generated run-metadata JSON were not available for verification. Provider updates and the absence of immutable model snapshots limit exact reproduction of new responses. Reproducing the numerical analyses from saved outcomes does not require regenerating responses.

## Collecting new responses

New collection incurs provider charges and may reach different model versions. Use Notebook 00 for implementation checks and the excluded P0 pilot, then Notebook 01 for collection. Do not use the P0 pilot to tune the confirmatory hypotheses or design.

Store credentials as environment variables or project secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, and `DEEPINFRA_API_KEY`. Qwen does not require a workspace ID in the current module. The Gemini client uses `GEMINI_API_KEY`.

Enable pilot calls deliberately with `RUN_TECHNICAL_PILOT_API_CALLS = True`. For confirmatory collection, the code additionally requires the preregistration reference, `PREREGISTRATION_FROZEN = True`, `CONFIRMATORY_UNLOCK=I_HAVE_PREREGISTERED`, and `RUN_CONFIRMATORY_API_CALLS = True`. Keep API keys and `.env` files out of version control.

## Anonymous review copy

This GitHub repository is associated with its owner's account and is not an anonymous review link. The existing `IRB/` materials also require an identity check before inclusion in a review copy. An anonymous archive should contain a separate snapshot of the relevant code and data, with identifying metadata and links checked, and an anonymous preregistration link or redacted registration document. No anonymous archive URL has been generated yet.
