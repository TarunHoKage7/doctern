# Doctern: Clinical Context Assistant

Doctern reviews a patient case, the doctor's proposed diagnosis and plan, and dated Indian clinical and surveillance evidence. It returns a short advisory: a plausible secondary diagnosis, discrepancies, why, suggested checks, targeted questions, and the exact source passages. The clinician makes the decision. All patient data in this repository is fictional.

Built at the Google DeepMind Hyderabad Hackathon, 26 September 2026. The full brief is in [plan/](plan/clinical-context-assistant-implementation-brief.md).

## Architecture

```
case JSON ──> validate_input ──> bind_snapshot (immutable, dated evidence)
          ──> Gemma 4: lookup plan (allowlisted local tools only; app validates + executes)
          ──> execute_local_lookups (SQLite FTS5 retrieval + source-backed medication rules)
          ──> Gemma 4: tentative assessment
          ──> MedGemma: independent assessment (does not see Gemma's conclusions)
          ──> reconcile_if_needed: one extra Gemma lookup, then MedGemma final specialist review
          ──> validate_result (citation IDs exist, case paths exist, rule matches never dropped)
          ──> persist_result (SQLite; state saved after every stage, resumable)
```

- **Application code owns the sequence.** Models choose lookups inside fixed limits: 3 initial lookups, 1 reconciliation lookup, 6 passages, 1 schema repair per call, 2 questions.
- **Validation checks traceability, not medical truth.** Unsupported conclusions are withheld and listed. A model failure or missing evidence can never produce "No material discrepancy identified in this review."
- **Result modes are labelled.** `live_local` is a fresh run. `cache_hit` is an identical earlier run, shown with its original time. `recorded_replay` is the saved output of an earlier genuine run.

## Models and runtimes

| Role | Model | Runtime |
|---|---|---|
| Orchestration, lookup selection, assessment | Gemma 4 31B instruct (`gemma-4-31b-it`) | Gemini API, hosted, **online** |
| Clinical review and final specialist review | MedGemma 4B instruct (`medgemma:4b`) | Ollama on loopback, **local** |

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

Run the contract tests with `python -m pytest -q tests`. They use mock model adapters and check the application logic only. They are not a real-model demonstration.

## Fictional cases

1. **Ravi** is NS1-positive with ibuprofen proposed for joint pain. The expected finding is a source-backed NSAID-in-dengue concern.
2. **Meera** has mild dengue plus diabetes, with a guideline-consistent plan. No discrepancy should be invented.
3. **Sanjay** is in Bengaluru Urban the week after the week 31 dengue outbreak, takes aspirin for coronary artery disease, and has unknown test results. A clarification creates a new revision and a fresh review.

## Not implemented

- **Voice intake.** The "Planned voice intake" panel shows the intended flow only.
- **Other inputs.** Image analysis, PDF/OCR of patient documents, and CT/MRI are not implemented. Attachments are reported as not analyzed.
- **Unattended updates.** There is no automatic weekly scheduler or source discovery. The refresh command is manual.
- **Validation.** This is not clinically validated and not a medical device.
