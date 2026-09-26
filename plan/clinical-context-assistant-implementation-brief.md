# Clinical Context Assistant — implementation handoff for Claude

This is an implementation request for a two-hour hackathon prototype. Read the entire brief before building, then implement the smallest complete path, verify it, and report what actually works. The teammate owns implementation; Tarun is preparing the presentation. One implementation owner should build both workflows in the same application.

The product assists doctors in India by reviewing a patient case, the doctor's proposed diagnosis and treatment, and relevant clinical and regional evidence. Use the exact user-facing term "Plausible secondary diagnosis". Suggestions are advisory; the clinician makes the clinical decision.

The selected direction is the local-agent problem statement. The demonstrated clinical loop must work with internet access disabled. Research refresh runs separately while connected. Voice intake is a future-flow illustration for this build. An optional Gemini Live interface is secondary and online; it must not become a dependency of the clinical loop. Do not substitute Gemini Flash for either local model.

## 1. Review conclusions and scope decisions

Gemma 4 is essential. It organizes the case, selects necessary lookup calls, interprets returned evidence, and produces its own tentative assessment. MedGemma is also essential. It reviews the medical information, evaluates plausible alternatives and discrepancies, and performs the final specialist review of any contested clinical conclusion.

Application code owns the sequence, tool permissions, retry limits, persistence, and output validation. Model choice of lookups happens inside that sequence. This is a controlled workflow with bounded model decisions.

Preserve disagreements. MedGemma has the final specialist review role, but neither its agreement with Gemma 4 nor its own confidence establishes clinical correctness. If evidence remains inadequate, show uncertainty or request one material clarification. An application can validate citation identity and quoted text; it cannot mechanically establish that a clinical interpretation is correct.

Keep these distinctions explicit:
- A medicine can be appropriate for an existing condition even if unrelated to the presenting symptom. Compare its stated indication, current conditions, and relevant evidence before raising a concern.
- Missing information is unknown. An unreported allergy is not "no allergies".
- Regional surveillance is context. A local outbreak does not establish the patient's diagnosis, and no local report does not exclude a disease.
- Doctor specialty tailors explanation and referral context. It does not determine which model or clinician is correct.
- Prior treatment records describe what happened. They are not verified treatment recommendations or training labels.
- A cached run demonstrates a recorded computation. It is labeled and never represented as a fresh model response.
- Local inference is a deployment choice, not evidence of clinical validation or legal compliance.

Build now:
- Four fictional doctor profiles, three fictional patient cases, direct JSON intake and validation.
- Gemma 4 orchestration and MedGemma clinical review through genuine local inference.
- A small, dated evidence snapshot, local retrieval, one repeatable research-refresh command.
- Concise advisory output with source passages, targeted clarification, and a recheck after editing a case.
- Persisted run states, genuine recorded replays, and visible source/model provenance.
- A usable local demo page plus a small explanation of the intended future voice flow.

Scope limits for the two-hour build:
- Focus clinical evidence on one demonstrable presentation, such as febrile illness, rather than attempting all specialties.
- Preserve attachment fields. Implement image analysis only if one supported image fixture and the actual vision adapter pass an early smoke check. Report unsupported attachments explicitly.
- CT/MRI volume processing, general PDF/OCR extraction, live voice, training, automatic national surveillance crawling, a knowledge graph, a full interaction checker, and a nationwide seasonality predictor are later work.
- A refresh command establishes the weekly update mechanism. An unattended weekly scheduler is optional later work.
- Use synthetic patient history for the demo. Public artifacts contain fictional records, not real patient data.
- Do not reduce the two-model core to a static chatbot to meet the time target.

## 2. Choose the shortest workable implementation

First inspect the implementation repository, its instructions, and available runtimes. Reuse an existing working stack. For an empty repository, use:
- Python with FastAPI, Pydantic, httpx, and SQLite.
- SQLite FTS5 plus small curated synonym mappings for retrieval. A deterministic local text-search fallback is acceptable if FTS5 is unavailable.
- One HTML/CSS/JavaScript frontend served by FastAPI, using ordinary polling for run progress.
- One application process and adapters to already available local model runtimes.
- Local JSON evidence records and SQLite run persistence. Avoid introducing a vector service or an agent framework.

A suggested layout:
```
app/
  main.py
  schemas.py
  workflow.py
  models.py
  tools.py
  evidence.py
  storage.py
  refresh.py
  static/index.html
  static/app.js
  static/styles.css
data/
  doctor_profiles.json
  cases/
  source_manifest.json
  snapshots/
  medication_rules.json
prompts/
  gemma_plan.txt
  gemma_assess.txt
  medgemma_assess.txt
  medgemma_review.txt
tests/
.env.example
README.md
```

Keep schemas authoritative in schemas.py and generate their JSON schemas for model output validation. The frontend consumes those contracts rather than inventing its own field names.

The two-hour estimate assumes access to the model weights, their terms have been accepted where required, and hardware can run them. Within the first ten minutes, obtain a genuine response from each model. Record runtime, model identity, quantization, and latency. If either cannot run, report that blocker immediately while completing independent UI/schema/retrieval work. A replay or stub must not be reported as a completed two-model implementation.

Prefer an available Gemma 4 E2B or E4B instruction model for orchestration and MedGemma 1.5 4B instruction model for clinical review. Use MedGemma 27B only if it is already provisioned and passes the memory/latency check. MedGemma 1.5 refers to the 4B release; record the actual model used.

Define two separate adapters:
- GemmaClient: structured planning/tool selection and assessment.
- MedGemmaClient: structured clinical review and, if supported, image input.

Both can target local HTTP endpoints or an existing local Transformers runtime. Verify the exact endpoint and chat template supported by the installed runtime. An OpenAI-compatible endpoint is a transport choice, not a reason to assume every runtime implements tools, JSON schema, or vision identically. If native tool calls are unavailable, Gemma may return a schema-validated tool-request object that the application dispatches.

Use explicit model configuration, per-call timeouts, bounded output lengths, and stable decoding settings. A fixed workflow does not make generated clinical answers fully deterministic. Record versions and settings for reproducibility.

## 3. Fix the shared data contracts before implementing the workflow

DoctorProfile is initialization data: profile_id, display_name, qualification, specialty, and practice_location with country, state, district, and town. Seed general physician, pediatrician, orthopedist, and cardiologist profiles with fictional Indian names. Do not infer clinical competence from a qualification label.

CaseRequest has an envelope and exactly three clinical sections: patient, presentation, and doctor_plan. The example below demonstrates shape, not a recommended treatment or a fully assessed case:
```json
{
  "schema_version": "1",
  "case_id": "demo-meera",
  "revision": 1,
  "doctor_profile_id": "general-physician",
  "encounter_at": "2026-09-26T10:00:00+05:30",
  "patient": {
    "display_name": "Meera",
    "age_years": 46,
    "sex": "female",
    "residence": {
      "country": "India",
      "state": "Telangana",
      "district": "Hyderabad"
    },
    "known_conditions": ["type 2 diabetes"],
    "current_medications": [
      {
        "name": "metformin",
        "dose": null,
        "route": null,
        "frequency": null,
        "indication": "type 2 diabetes"
      }
    ],
    "allergies": null,
    "family_history": ["mother has hypertension"],
    "pregnancy_status": "unknown"
  },
  "presentation": {
    "symptoms": [
      {
        "name": "fever",
        "duration_days": 3,
        "severity": null
      },
      {
        "name": "body aches",
        "duration_days": 3,
        "severity": null
      }
    ],
    "vitals": null,
    "test_results": [],
    "recent_travel_or_exposure": null,
    "attachments": [],
    "notes": null
  },
  "doctor_plan": {
    "proposed_diagnosis": "viral illness",
    "proposed_treatment": null,
    "proposed_medications": [],
    "rationale": null
  }
}
```

Additional typing rules:
- Use null for unknown and an empty list only for explicitly reported absence.
- Each allergy includes substance, reaction, and severity when known.
- Current and proposed medication entries use the same typed structure, including dose value/unit, route, frequency, and intended indication when supplied.
- Tests and vitals include value, unit, and measurement time when available.
- Attachments refer to server-managed file IDs with media type and modality. Resolve them inside a fixed local directory; never execute arbitrary paths supplied in JSON.
- Require a valid doctor reference and at least one symptom or clinical question. Unknown age, allergy status, etc. are represented explicitly rather than guessed. Clinical missing-data questions are decided by material relevance.
- Patient residence/travel may differ from the doctor's location. Retrieval uses patient context when supplied, with any location fallback visible.
- A doctor's clarification creates a new immutable revision and invalidates the old review.

EvidenceRecord fields:
```
evidence_id, snapshot_id, source_kind, title, publisher, source_url,
published_at, retrieved_at, reporting_period_start, reporting_period_end,
geography, condition_tags, medication_tags, jurisdiction,
exact_excerpt, document_locator, content_sha256, local_document_id
```
Dates may be unknown; do not replace publication dates with download dates. document_locator identifies a section or PDF page. source_kind distinguishes surveillance, guideline, drug_label, regulatory_notice, seasonal_reference, and ayush_research. Model summaries are stored separately and retain evidence IDs.

ModelAssessment fields:
```
candidate_conditions, candidate_discrepancies, missing_material_facts,
supporting_evidence_refs, conflicting_evidence_refs, unresolved_questions
```
Each candidate points to actual case field paths and evidence IDs. Restrict the fixture vocabulary enough to compare structured issue types, such as medication_allergy, guideline_conflict, or missing_required_measurement. Do not implement general diagnostic equivalence using naive string equality.

ReviewResult separates clinical disposition from execution status:
- Clinical disposition: material_concern, no_material_discrepancy_identified, needs_clarification, or insufficient_evidence.
- User-facing sections: "Plausible secondary diagnosis", "Discrepancies", "Why", "Suggested checks", "Questions", and expandable "Sources".
- At most two plausible secondary diagnoses, three material discrepancies, and two targeted questions. Empty sections remain hidden.
- Every medical concern and suggested check includes supporting case facts, an explanation, and valid evidence references. Plausibility remains qualified; avoid uncalibrated percentages.
- Preserve unresolved disagreements and relevant evidence limitations.
- Execution metadata: run_id, case_revision, snapshot_id, model identities, prompt/config versions, timestamps, durations, mode, and technical status.
- mode is live_local, cache_hit, or recorded_replay. Illustrative future UI is not a clinical run mode.

The no-discrepancy message is "No material discrepancy identified in this review." It is not approval of the diagnosis, prescription, or completeness of care. Evidence or runtime failure must never produce that status.

## 4. Research workflow: prepare a reusable local snapshot

This workflow is logically independent from patient review, even though one developer implements both.

Implement a source-manifest-driven refresh command. It fetches a small set of explicitly configured official source URLs, extracts usable text, preserves attribution and document locations, indexes records, and atomically publishes a new immutable snapshot. A failed refresh leaves the last usable snapshot intact and reports the failed sources.

For this prototype, seed a small reviewed evidence pack in the repository. The refresh command can update or reindex those manifest entries. Discovering all newly published weekly reports automatically is optional; document any manual manifest update.

Gemma 4 also prepares the weekly research brief: after ingestion, it selects relevant local lookups and produces a short snapshot_brief.json containing dated regional observations, any separately sourced seasonal context, coverage gaps, and evidence IDs for every claim. Keep that planning/lookup cycle bounded to the same small tool budget used for cases. If the brief includes medical implications, MedGemma reviews those statements before they are included; unsupported implications remain unresolved or are omitted. Store each actual model response and the brief's review status. The snapshot's original passages remain authoritative. Patient review can retrieve both the brief and its cited passages, with summaries never replacing the underlying evidence. Precompute this real research run once for the demo; later refreshes repeat it against new records.

Prioritize:
- IDSP/NCDC weekly outbreak reports for dated geographic context.
- ICMR or official disease-program guidelines for clinical reference.
- CDSCO notices and accessible medicine labeling for the limited medication checks.
- National Formulary of India material only if legitimately available to the team. Its publication listing is not a free machine-readable clinical API.
- An authoritative seasonal reference if historical seasonality is shown. A single outbreak report cannot establish a seasonal trend.

Use a real historical report with its actual reporting period if current local data is unavailable. A fictional case may be explicitly set in that historical period. Never portray an old report as a current local outbreak.

Keep AYUSH sources separately typed. Include only when relevant to the case or recorded medicine use, preserving the system of medicine, publication type, and uncertainty. Avoid blending unlike source types into one undifferentiated authority score.

Return surveillance coverage limits visibly: a missing district report is missing coverage; an old report is dated context. Do not convert outbreak case counts into individual diagnostic probabilities.

Build these local tools:
- search_evidence(query, source_kinds, patient_location, encounter_at, limit).
- get_evidence(evidence_ids).
- check_medication_facts(proposed_medications, relevant_patient_facts).

The medication tool performs only the small, source-backed checks implemented in the evidence pack. It returns matched rules, contradictions, and unknowns. It does not certify comprehensive drug safety, infer that US approval means Indian approval, or generate replacement prescriptions.

A general medication rule contains a typed condition, required patient facts, an evidence reference, and a supported warning. Evaluate declared operators with ordinary code; never execute model-generated expressions. Rules are based on clinical facts and cited guidance, not demo names or case IDs.

Keep the public evidence cache separate from case memory. Synthetic encounter logs can demonstrate history and simple counts. Any aggregated history is labeled as local observations with a denominator and time window; it does not establish correct care or regional prevalence.

## 5. Per-case workflow: application-controlled and bounded

Use this order and save state after every completed stage:

validate_input → bind_snapshot → gemma_lookup_plan → execute_local_lookups →
gemma_assessment → medgemma_assessment → reconcile_if_needed →
validate_result → persist_result.

Gemma's lookup plan chooses only the local tools from the allowlist. The application validates names, types, case references, and limits before execution. Source content and case notes are data, never executable instructions.

Gemma's assessment receives the actual case and retrieved passages. It may propose candidate discrepancies or plausible alternatives, along with specific missing facts. Keep its concise evidence rationale, not an exposed chain-of-thought trace.

MedGemma's first assessment receives the same case and evidence, initially without Gemma's candidate conclusions. It supplies its own medical interpretation. The doctor's proposed diagnosis and medicines remain part of the case because reviewing them is the task.

If there is disagreement or a potentially material concern, allow one bounded reconciliation:
1. Gemma may request a targeted additional lookup about the specific unresolved issue.
2. Execute it against the local snapshot.
3. MedGemma performs the final specialist review using both short assessments and the additional evidence.
4. Retain unresolved disagreement; do not keep asking until the models agree.

Starting limits, configurable and recorded:
- At most three Gemma lookup requests initially and one extra during reconciliation.
- At most six passages in a review context, trimmed without losing relevant caveats.
- One MedGemma initial pass and one optional final review.
- One schema-repair attempt per malformed model response.
- At most two clarification questions per turn.
- A per-call timeout and a total run budget derived from the startup smoke measurements; start with 60 seconds and 180 seconds respectively, adjust visibly if the hardware requires it.

Token caps and low-temperature settings control cost and latency. If memory permits, warm both local models before the demonstration. Sequential execution is the default; parallel model loading or inference is optional after confirming available memory.

The application then checks:
- All cited evidence IDs exist in the bound snapshot.
- Referenced case fields and quoted facts exist in the submitted revision.
- Any excerpt used is genuinely from the source and its location is available.
- A discrepancy has patient-specific relevance, including the proposed medicine's intended indication.
- A potential secondary diagnosis is supported by case facts, with missing or conflicting facts retained.
- A source-only occurrence or two-model agreement is not the sole reason for an alert.

These checks validate structure and traceability, not medical truth. Source applicability remains an explicitly qualified clinical assessment. Unsupported conclusions are withheld from the definitive discrepancy list and surfaced as uncertainty when material.

Ask a question only when its answer could change a current concern, interpretation, or suggested check. Explain why the requested fact matters. Optional missing family history should not automatically block every case. Urgent source-backed red flags can be raised without claiming the doctor's diagnosis has been disproved.

If evidence is absent, the models time out, or a response remains invalid, preserve a partial/failed state and explain the limitation. Never fill the result with canned medical conclusions.

## 6. Cache, persistence, and restart behavior

Use SQLite for runs, stage records, doctor decisions, and references to immutable evidence snapshots. Store a concise event trail with stage, tool/model identity, timing, and outcome.

Derive a result cache key from canonicalized case JSON, doctor profile content, attachment content hashes, snapshot ID, both model identifiers and revisions/quantization, generation configuration, schema version, prompt versions, and rule version.

A changed case field, source snapshot, prompt, model, or attachment invalidates reuse. Bind the snapshot once at run creation so a background refresh cannot change evidence halfway through the review.

Cache only completed runs. Recorded replays must come from actual model executions, with their original input, evidence, timestamps, and raw validated model responses. An illustrative fixture stays visibly illustrative. A cache hit displays the original generation time and returns immediately; it is not a "live" response.

Use a client request ID to prevent duplicate runs on repeated clicks. Resume an interrupted run from the last completed stage; rerun an uncertain in-flight model call rather than assuming success. Never reuse prior conclusions after clarification changes the case.

Keep lightweight local logs and exclude secrets, credentials, real patient data, and model weights from public repository output. Serve the app on loopback by default.

## 7. Local API and presentation

Minimal API:
- GET /api/profiles — fictional doctor profiles.
- GET /api/cases — available synthetic fixtures.
- GET /api/evidence/status — snapshot date, scope, and source coverage.
- POST /api/reviews — validated case plus mode and client request ID; returns run_id.
- GET /api/reviews/{run_id} — progress and eventual result.
- POST /api/reviews/{run_id}/clarifications — typed answers; creates a revised case/run.
- POST /api/reviews/{run_id}/resume — continue an interrupted run.
- GET /api/evidence/{evidence_id} — cited local passage and provenance.
- GET /api/health — app and model readiness, excluding credentials.

Use a single clean clinician page:
1. Doctor profile selector and a case selector.
2. Editable JSON with a short human-readable case preview.
3. Review action and genuine progress stages: gathering evidence, Gemma 4 assessment, MedGemma review, result.
4. Compact advisory card and expandable source passages.
5. Clarification inputs, edit/recheck, and doctor response: acknowledge, dismiss with a note, or request another check.
6. A visible Live local / Cached result / Recorded replay label and the evidence snapshot date.

Doctor actions record their decision; they do not automatically update model weights or mark a diagnosis as proven.

Place an optional technical details panel below the clinical content for the demo: model identities, actual calls, timing, snapshot ID, and why the result was cached. Avoid flooding the primary doctor screen with implementation details.

The landing/pitch section explains the product and shows the future flow:
speech → structured case → material clarification → clinical review.
Label it "Planned voice intake". The working prototype consumes JSON. A displayed microphone must not imply an implemented listening feature.

Optional images: render a local thumbnail and indicate whether the configured model actually processed it. Do not treat an attached filename as analyzed imaging. If image support is incomplete, show "Attachment not analyzed" and allow the text review to continue with that limitation.

Optional Gemini Live comes only after the complete offline path passes. It is an online voice feature and uses fictional demo content. It must be removable without changing the core workflow.

## 8. Model task definitions

Keep these responsibilities in separate versioned prompt files, with explicit JSON output schemas. The following are task contracts, not requirements to reproduce hidden reasoning.

Gemma planning:
Read the structured case and available local tools. Identify which evidence is needed to examine the doctor's proposal. Return bounded lookup requests, necessary patient fact references, and any clearly material missing fields. Preserve unknowns.

Gemma assessment:
Use the case and supplied evidence passages to form a tentative clinical assessment. Return a small set of candidate discrepancies or plausible alternative conditions only when they materially affect this case. Link the reasoning summary to case facts and source IDs. Distinguish lack of evidence from contradiction.

MedGemma assessment:
Review the case, doctor's plan, and supplied evidence independently. Identify medically plausible alternatives, supported concerns, conflicting information, and material missing data. Provide a concise clinical rationale with evidence references. Treat images as unavailable unless they were actually supplied through the supported vision path.

MedGemma final review:
Examine the two assessments and additional evidence for the specific disputed issue. State whether the concern is supported, unsupported, or unresolved, and why. Preserve unresolved uncertainty. Produce the clinical fields of the advisory result for deterministic validation and display.

## 9. Fixtures and meaningful verification

Create three fictional cases in the selected scope:
- A proposed plan with a clear source-backed concern, such as a recorded serious medicine allergy conflicting with a proposed medicine, or a proposed NSAID in a case explicitly under review for dengue. Verify the relevant official passage and applicability before seeding an expected finding.
- A related case with no supported discrepancy in the available evidence. Confirm the UI does not invent an alternative diagnosis to fill a section.
- An incomplete case in which a specific missing fact changes whether the concern applies. Confirm a targeted clarification produces a new revision and fresh review.

Use a genuine historical surveillance report for the regional context demo, clearly dated. Do not invent a local outbreak to make the example convincing. A clinical guideline and a matching source record are sufficient for a first run if district coverage is missing.

Clinical examples are demonstration cases, not evidence of diagnostic accuracy. Compare exact behavior and source support rather than reporting an accuracy percentage from three examples.

Automated contract checks should cover:
- Unknown versus explicitly absent data.
- Rejection of arbitrary tool names and invalid arguments.
- Invalid citation IDs and fabricated case references.
- Missing evidence or model failure cannot become "no discrepancy".
- Any relevant case/source/model change misses the result cache.
- Repeated submission is idempotent and interrupted runs resume.
- Model disagreement reaches the bounded unresolved state.
- Fixture names and case IDs are not used as diagnostic triggers.

End-to-end verification:
- Run at least one genuine Gemma 4 + MedGemma case; retain provenance.
- Change one clinically relevant field and demonstrate a fresh run.
- Stop/restart the app and resume or reopen the saved case.
- Disable internet access or run in a network-restricted environment while preserving loopback model access. Complete the local review and source display.
- Inspect the actual UI once to verify that result status, citations, cache label, and errors are readable.
- Record actual elapsed times and which image/voice capabilities were exercised.

Mock adapters are appropriate for deterministic failure tests. Report them as tests, never as a successfully completed real-model demo.

## 10. Two-hour execution order

Minutes 0-10:
Inspect the repository, check hardware/runtime and weights, obtain one genuine response from each model, and choose exact adapters. Finish when both model identities and measured latency are known, or the specific blocker is reported.

Minutes 10-25:
Freeze Pydantic contracts, seed fictional profiles/cases, and prepare a small evidence snapshot with valid source locations. Produce the brief with Gemma 4 and have MedGemma review any medical implications using the same model adapters. Finish when a case validates, local lookup returns a traceable passage, and the research brief has recorded provenance.

Minutes 25-65:
Implement the fixed workflow, actual Gemma tool selection, MedGemma assessment, bounded reconciliation, schema validation, and source checks. Finish when one real case runs end to end through both models.

Minutes 65-90:
Connect the clinician page, evidence panel, clarification/recheck, persistence, and exact cache/replay modes. Finish when a user can complete the case flow through the page.

Minutes 90-110:
Run the behavioral checks, offline execution, edited-case recheck, and restart test. Fix failures affecting source integrity or the real two-model loop first.

Minutes 110-120:
Record an authentic successful run for replay, document commands/configuration, capture the demo sequence, and produce a concise capability report.

If time slips, cut in this order: Gemini Live integration, image processing, automatic source discovery, additional doctor specialties, UI animation. Retain the actual two-model text loop, source traceability, honest result status, and direct JSON input.

## 11. Completion report and deliverables

Deliver the working repository with:
- One documented setup/start path for the actual OS and runtime.
- .env.example containing placeholders and both model configurations.
- A refresh/reindex command and manifest, with actual snapshot sources and dates.
- Three synthetic cases and at least one genuine recorded two-model replay.
- Direct JSON review, citations, clarification, and case recheck.
- Contract test results, one real-model end-to-end result, and an offline verification result.
- A README distinguishing implemented capabilities, illustrative future flow, and unavailable features.

The final implementation report must state exact models/runtimes, how to launch, observed latency, what was verified live, what is replayed, and any remaining blocker. Do not claim local multimodal processing, weekly automated refresh, successful MedGemma review, or offline operation without corresponding execution evidence. Leave publication/deployment to the team unless separately authorized.

## References and what they establish

- [Gemma 4 tool calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
  Gemma selects tool calls; the application validates and executes them.
- [MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
  Model variant, intended developer use, and need for independent clinical validation.
- [Local server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
  One possible serving transport; support must be confirmed for the exact build and model.
- [IDSP weekly reports](https://www.idsp.mohfw.gov.in/index4.php?lang=1&level=0&lid=3689&linkid=406)
  Official dated outbreak publications.
- [Example historical IDSP report, March 2-8, 2026](https://idsp.mohfw.gov.in/WriteReadData/l892s/7390466771776672948.pdf)
  A historical fixture candidate, not current September surveillance.
- [ICMR workflows](https://www.icmr.gov.in/standard-treatment-workflows-stws)
  Clinical reference material; preserve each document's edition and source location.
- [NCVBDC dengue guideline, 2023](https://ncvbdc.mohfw.gov.in/Doc/National%20Guidelines%20for%20Clinical%20Management%20of%20Dengue%20Fever%202023.pdf)
  Possible source for a tightly scoped clinical demonstration.
- [CDSCO](https://www.cdsco.gov.in/opencms/opencms/en/)
  Indian drug regulatory information; approval is distinct from patient-specific suitability.
- [National Formulary publication access](https://ipc.gov.in/about-us/departments/publication.html)
  Check legitimate access and reuse terms before ingestion or public redistribution.
- [AYUSH Research Portal](https://arp.ayush.gov.in/)
  Optional, separately labeled source family.
