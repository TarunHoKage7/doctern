# Doctern: Clinical Context Assistant

Doctern reviews a patient case, the doctor's proposed diagnosis and plan, and dated Indian clinical and surveillance evidence. It returns a short advisory: a plausible secondary diagnosis, discrepancies, why, suggested checks, targeted questions, and the exact source passages. The clinician makes the decision. All patient data in this repository is fictional.

Built at the Google DeepMind Hyderabad Hackathon, 26 September 2026. The full brief is in [plan/](plan/clinical-context-assistant-implementation-brief.md).

> **Demo note.** This hackathon demo calls Gemma 4 through the online Gemini API. MedGemma runs locally, or optionally on Vertex AI. The product is designed as a fully local setup: set `GEMMA_BACKEND=ollama` to run Gemma 4 E2B or E4B on the same machine, and no patient data leaves it.

## Architecture

```
case JSON ──> validate_input ──> bind_snapshot (immutable, dated evidence)
          ──> Gemma 4: lookup plan (allowlisted local tools only; app validates + executes)
          ──> execute_local_lookups (SQLite FTS5 retrieval + source-backed medication rules)
          ──> Gemma 4: tentative assessment
          ──> MedGemma: independent assessment (does not see Gemma's conclusions)
          ──> reconcile_if_needed: one extra Gemma lookup, then MedGemma final specialist review
          ──> verify_result: a fresh Gemma 4 chat checks the advice against the case, what each model
              actually said, the passages and the weekly summary; it removes unsupported items and
              corrects any false account of the disagreement
          ──> validate_result (citation IDs exist, case paths exist, rule matches never dropped)
          ──> persist_result (SQLite; state saved after every stage, resumable)
```

- **Application code owns the sequence.** Models choose lookups inside fixed limits: 3 initial lookups, 1 reconciliation lookup, 6 passages, 1 schema repair per call, 2 questions.
- **Validation checks traceability, not medical truth.** Unsupported conclusions are withheld and listed. A model failure or missing evidence can never produce "No material discrepancy identified in this review."
- **Gemma-only fallback.** If MedGemma is unreachable or returns invalid output, Gemma 4 finishes the review. The result is labelled "Gemma 4 only: MedGemma specialist review unavailable", and the limitation is listed first. A fallback review can never report "No material discrepancy identified." Verified live on Ravi's case with MedGemma disabled: it completed in 51 s and flagged the ibuprofen concern.
- **Replays are locked to their case.** A recorded replay plays only if the case is unchanged since recording. An edited case gets a live review.
- **Result modes are labelled.** `live_local` is a fresh run. `cache_hit` is an identical earlier run, shown with its original time. `recorded_replay` is the saved output of an earlier genuine run.

## Models and runtimes

| Role | Model | Runtime |
|---|---|---|
| Orchestration, lookup selection, assessment | Gemma 4 31B instruct (`gemma-4-31b-it`) | Gemini API, hosted, **online** |
| Clinical review and final specialist review | MedGemma 4B instruct (`medgemma:4b`) | Ollama on loopback, **local** (default) |
| Same role, optional | MedGemma 27B (`medgemma-27b-it-dicom`) | Vertex AI Model Garden endpoint, **online**. Set `MEDGEMMA_BACKEND=vertex`. |

The team chose hosted Gemma 4 31B for orchestration because it is the strongest Gemma 4 available. **The clinical loop therefore needs internet for the Gemma steps.** Setting `GEMMA_BACKEND=ollama` and `GEMMA_MODEL=gemma4:e2b` (or `gemma4:e4b`) switches orchestration to local Gemma 4. Offline operation of that configuration has not been verified in this build.

## Evidence snapshot

Run `python -m app.refresh` to build a new immutable snapshot from [data/source_manifest.json](data/source_manifest.json). Passages are cut verbatim from the official PDFs by anchor phrases. A failed source is reported and never replaces the last good snapshot.

- **Guideline.** NCVBDC *National Guidelines for Clinical Management of Dengue Fever 2023*: 8 passages with PDF page locators.
- **Surveillance.** IDSP *Weekly Outbreak Report, week 31 2026* (27 July to 2 August 2026): reporting status, the Bengaluru Urban dengue outbreak entry, and the COVID-19 status note.
- **Coverage limits.** A missing district report means missing coverage. Surveillance is dated context and never a diagnosis.

Medication checks in [data/medication_rules.json](data/medication_rules.json) are a small set of source-backed rules, such as an NSAID proposed while dengue is under consideration. They are not a comprehensive interaction checker.

## Run it (Windows, tested)

```bash
pip install -r requirements.txt
copy .env.example .env        # set GEMINI_API_KEY
ollama pull medgemma:4b
ollama serve
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Pick a doctor and a case, then click **Review case**.

Run one case from the command line with `python run_case.py data/cases/01-ravi-nsaid-dengue.json`.

Run the contract tests with `python -m pytest -q tests`. They use mock model adapters and check the application logic only. They are not a real-model demonstration.

## Fictional cases

1. **Ravi** is NS1-positive with ibuprofen proposed for joint pain. The expected finding is a source-backed NSAID-in-dengue concern.
2. **Meera** has mild dengue plus diabetes, with a guideline-consistent plan. No discrepancy should be invented.
3. **Sanjay** is in Bengaluru Urban the week after the week 31 dengue outbreak, takes aspirin for coronary artery disease, and has unknown test results. A clarification creates a new revision and a fresh review.

## What was verified on 26 September 2026

These were genuine runs with hosted Gemma 4 31B and local MedGemma 4B, Q4_K_M quantization, on an RTX 5050 laptop GPU with 8 GB.

| Case | Result | Wall time |
|---|---|---|
| Ravi | Material concern: ibuprofen with a positive NS1 test, citing the NCVBDC passage. It needed one resume after a truncated MedGemma output. | about 90 s |
| Meera | No material discrepancy identified. Nothing was invented. | 17 s |
| Sanjay | Material concern about continuing aspirin. Dengue was raised as a plausible secondary diagnosis using the dated week 31 Bengaluru Urban outbreak entry. It asked for the platelet count. | 49 s |

- **Latency.** A warm MedGemma 4B call takes about 2 to 17 s. The first call loads the model and takes about 66 s. Each Gemma 4 31B call takes about 4 to 13 s.
- **Quality.** MedGemma 4B output has visible errors. It overstates aspirin as "contraindicated", suggests shock without supporting findings, and garbles the disagreement text. The validation layer checks citations and case paths. It cannot check clinical correctness.
- **Replay.** Ravi's run is saved as a recorded replay in [data/replays/demo-ravi.json](data/replays/demo-ravi.json).
- **Not verified.** Offline operation was not tested, because Gemma runs through the hosted API. The Vertex MedGemma 27B adapter is written but has not been called yet.
- **Tests.** The contract tests pass. They use mock adapters.

## Advisory use

Doctern is an advisory tool. Every output is a suggestion with its sources, and the clinician makes the clinical decision. It does not claim clinical validation, regulatory approval, or correctness of any advisory output.

## Weekly research summary

Run `python -m app.refresh --brief` to rebuild the evidence pack and write the weekly summary. Gemma 4 picks what to look up and writes dated observations, each citing its passages. It also lists coverage gaps, such as how many states reported that week. Statements with clinical implications go to MedGemma, which marks each one supported, unsupported, or unresolved. Unsupported statements are dropped. The summary is saved next to the pack, shown on the page, and passed to both models during a case review. The original passages always take priority.

## Full product scope and current status

| Capability | Status in this build |
|---|---|
| Two-model case review with cited evidence | Working |
| Weekly research summary | Working (`--brief`) |
| Clarification and recheck | Working |
| Gemma-only fallback | Working |
| Voice intake with Gemini Live | In scope, next build. The page shows the planned flow. |
| Image analysis with MedGemma DICOM | In scope, next build. Radiology report text is used today. |
| Automatic weekly crawling of IDSP reports | In scope, next build. The manifest is updated by hand today. |
| Full drug interaction checker | In scope, next build. A small set of source-backed rules runs today. |
| National seasonality view | In scope, next build. |

## COVID-19 versus common cold case

Arjun's fictional case has COVID-like symptoms, a household COVID contact, and chest X-ray report text. The doctor's plan is "common cold" with a return to work tomorrow. In a live run on 26 September 2026, Doctern returned a material concern:

- **Plausible secondary diagnosis.** COVID-19.
- **Discrepancy.** COVID-19 was not considered despite the exposure and the imaging text.
- **Contagion warning.** A return to work contradicts isolation needs.
- **Suggested check.** A COVID-19 RT-PCR or rapid antigen test.

That run was Gemma-only, because MedGemma timed out, and the page labels it that way. Its only citation is the IDSP COVID status note, which does not support the clinical advice. The official ministry and ICMR COVID guideline PDFs could not be downloaded from the build machine. Adding that guideline to [data/source_manifest.json](data/source_manifest.json) gives the precautions a proper source.

