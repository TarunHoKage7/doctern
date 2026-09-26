"""Authoritative data contracts. Model output schemas are generated from these classes.

Convention: None means UNKNOWN. An empty list means EXPLICITLY REPORTED ABSENT.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- profiles
class Location(Strict):
    country: str
    state: Optional[str] = None
    district: Optional[str] = None
    town: Optional[str] = None


class DoctorProfile(Strict):
    profile_id: str
    display_name: str
    qualification: str
    specialty: str
    practice_location: Location


# ---------------------------------------------------------------- case
class Quantity(Strict):
    value: Optional[float] = None
    unit: Optional[str] = None


class Medication(Strict):
    name: str
    dose: Optional[Quantity] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    indication: Optional[str] = None


class Allergy(Strict):
    substance: str
    reaction: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None


class Residence(Strict):
    country: str
    state: Optional[str] = None
    district: Optional[str] = None


class Patient(Strict):
    display_name: str
    age_years: Optional[float] = None
    sex: Optional[Literal["female", "male", "other"]] = None
    residence: Optional[Residence] = None
    known_conditions: Optional[list[str]] = None
    current_medications: Optional[list[Medication]] = None
    allergies: Optional[list[Allergy]] = None
    family_history: Optional[list[str]] = None
    pregnancy_status: Optional[Literal["pregnant", "not_pregnant", "unknown", "not_applicable"]] = None


class Symptom(Strict):
    name: str
    duration_days: Optional[float] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None


class Measurement(Strict):
    name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    measured_at: Optional[str] = None


class Attachment(Strict):
    file_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    media_type: str
    modality: Optional[str] = None


class Presentation(Strict):
    symptoms: list[Symptom] = Field(default_factory=list)
    clinical_question: Optional[str] = None
    vitals: Optional[list[Measurement]] = None
    test_results: Optional[list[Measurement]] = None
    recent_travel_or_exposure: Optional[str] = None
    attachments: list[Attachment] = Field(default_factory=list)
    notes: Optional[str] = None


class DoctorPlan(Strict):
    proposed_diagnosis: Optional[str] = None
    proposed_treatment: Optional[str] = None
    proposed_medications: Optional[list[Medication]] = None
    rationale: Optional[str] = None


class CaseRequest(Strict):
    schema_version: Literal["1"] = "1"
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    revision: int = Field(ge=1)
    doctor_profile_id: str
    encounter_at: str
    patient: Patient
    presentation: Presentation
    doctor_plan: DoctorPlan

    @model_validator(mode="after")
    def needs_symptom_or_question(self):
        if not self.presentation.symptoms and not self.presentation.clinical_question:
            raise ValueError("case needs at least one symptom or a clinical question")
        return self


# ---------------------------------------------------------------- evidence
SourceKind = Literal["surveillance", "guideline", "drug_label", "regulatory_notice",
                     "seasonal_reference", "ayush_research"]


class EvidenceRecord(Strict):
    evidence_id: str
    snapshot_id: str
    source_kind: SourceKind
    title: str
    publisher: str
    source_url: str
    published_at: Optional[str] = None
    retrieved_at: str
    reporting_period_start: Optional[str] = None
    reporting_period_end: Optional[str] = None
    geography: Optional[str] = None
    condition_tags: list[str] = Field(default_factory=list)
    medication_tags: list[str] = Field(default_factory=list)
    jurisdiction: str
    exact_excerpt: str
    document_locator: str
    content_sha256: str
    local_document_id: str


# ---------------------------------------------------------------- model outputs
ToolName = Literal["search_evidence", "get_evidence", "check_medication_facts"]
IssueType = Literal["medication_allergy", "guideline_conflict", "missing_required_measurement",
                    "risk_factor_not_addressed", "alternative_diagnosis_not_excluded", "other"]


class LookupRequest(Strict):
    tool: ToolName
    query: str = Field(max_length=200)
    source_kinds: list[SourceKind] = Field(default_factory=list)
    reason: str = Field(max_length=300)


class LookupPlan(Strict):
    lookups: list[LookupRequest] = Field(max_length=4)
    material_missing_fields: list[str] = Field(default_factory=list, max_length=4)


class CandidateCondition(Strict):
    condition: str
    case_fact_paths: list[str]
    evidence_ids: list[str]
    rationale: str = Field(max_length=500)


class CandidateDiscrepancy(Strict):
    issue_type: IssueType
    summary: str = Field(max_length=300)
    case_fact_paths: list[str]
    evidence_ids: list[str]
    rationale: str = Field(max_length=500)


class MissingFact(Strict):
    field_path: str
    why_it_matters: str = Field(max_length=300)


class ModelAssessment(Strict):
    candidate_conditions: list[CandidateCondition] = Field(default_factory=list, max_length=3)
    candidate_discrepancies: list[CandidateDiscrepancy] = Field(default_factory=list, max_length=4)
    missing_material_facts: list[MissingFact] = Field(default_factory=list, max_length=3)
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    conflicting_evidence_refs: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=3)


Disposition = Literal["material_concern", "no_material_discrepancy_identified",
                      "needs_clarification", "insufficient_evidence"]


class ReviewDiscrepancy(Strict):
    issue_type: IssueType
    summary: str
    why: str
    case_fact_paths: list[str]
    evidence_ids: list[str]
    status: Literal["supported", "unsupported", "unresolved"]


class SecondaryDiagnosis(Strict):
    condition: str
    why: str
    case_fact_paths: list[str]
    evidence_ids: list[str]


class SuggestedCheck(Strict):
    check: str
    why: str
    evidence_ids: list[str]


class Question(Strict):
    field_path: str
    question: str
    why_it_matters: str


class FinalReview(Strict):
    disposition: Disposition
    secondary_diagnoses: list[SecondaryDiagnosis] = Field(default_factory=list, max_length=2)
    discrepancies: list[ReviewDiscrepancy] = Field(default_factory=list, max_length=3)
    suggested_checks: list[SuggestedCheck] = Field(default_factory=list, max_length=3)
    questions: list[Question] = Field(default_factory=list, max_length=2)
    disagreements: list[str] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=4)


# ---------------------------------------------------------------- API envelopes
class ReviewRequest(Strict):
    case: CaseRequest
    mode: Literal["auto", "live_local", "recorded_replay"] = "auto"
    client_request_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,80}$")


class ClarificationAnswer(Strict):
    field_path: str
    value: object


class ClarificationRequest(Strict):
    answers: list[ClarificationAnswer]
    client_request_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{6,80}$")


class DoctorDecision(Strict):
    action: Literal["acknowledge", "dismiss", "request_check"]
    note: Optional[str] = Field(default=None, max_length=1000)


NO_DISCREPANCY_MESSAGE = "No material discrepancy identified in this review."
