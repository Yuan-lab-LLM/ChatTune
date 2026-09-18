#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Any, Callable


class ResponseValidationError(ValueError):
    pass


def json_documents(text: str) -> list[Any]:
    """Return JSON values from a plain JSON response or a streamed text response."""
    stripped = text.strip()
    if not stripped:
        raise ResponseValidationError("response body is empty")

    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    documents: list[Any] = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        documents.append(value)

    if not documents:
        raise ResponseValidationError("response does not contain a valid JSON value")
    return documents


def nested_value(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def contains_key(value: Any, expected: str) -> bool:
    if isinstance(value, dict):
        return expected in value or any(
            contains_key(item, expected) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_key(item, expected) for item in value)
    return False


def select_document(
    documents: list[Any], predicate: Callable[[Any], bool], description: str
) -> Any:
    for document in reversed(documents):
        try:
            if predicate(document):
                return document
        except (KeyError, TypeError):
            continue
    raise ResponseValidationError(f"no JSON document contains {description}")


def validate_basic_medical_record(documents: list[Any]) -> None:
    select_document(
        documents,
        lambda value: contains_key(value, "chief_complaint"),
        "chief_complaint",
    )


def validate_nonempty_json(documents: list[Any]) -> None:
    select_document(documents, lambda value: bool(value), "a non-empty JSON value")


def validate_inpatient(documents: list[Any]) -> None:
    select_document(
        documents,
        lambda value: isinstance(nested_value(value, "output", "inpatient"), dict),
        "output.inpatient",
    )


def validate_followup_questionnaire(documents: list[Any]) -> None:
    select_document(
        documents,
        lambda value: bool(nested_value(value, "output", "questionsList")),
        "a non-empty output.questionsList",
    )


def validate_followup_chat(documents: list[Any]) -> None:
    select_document(
        documents,
        lambda value: bool(nested_value(value, "output", "chat")),
        "a non-empty output.chat",
    )


def validate_followup_report(documents: list[Any]) -> None:
    def has_answers(value: Any) -> bool:
        questions = nested_value(value, "output", "questionsList")
        return isinstance(questions, list) and any(
            isinstance(question, dict) and bool(question.get("questionAnswer"))
            for question in questions
        )

    select_document(documents, has_answers, "questionnaire answers")


def validate_quality(documents: list[Any]) -> None:
    def has_quality_fields(value: Any) -> bool:
        output = nested_value(value, "output")
        return (
            isinstance(output, dict)
            and isinstance(output.get("basic_medical_record"), dict)
            and "control_quality" in output
        )

    select_document(
        documents,
        has_quality_fields,
        "output.basic_medical_record and output.control_quality",
    )


def validate_case2chat(documents: list[Any]) -> None:
    select_document(
        documents,
        lambda value: isinstance(value, dict)
        and bool(value)
        and isinstance(value.get("historical_conversations"), list),
        "historical_conversations",
    )


VALIDATORS: dict[str, Callable[[list[Any]], None]] = {
    "basic-medical-record": validate_basic_medical_record,
    "nonempty-json": validate_nonempty_json,
    "inpatient": validate_inpatient,
    "followup-questionnaire": validate_followup_questionnaire,
    "followup-chat": validate_followup_chat,
    "followup-report": validate_followup_report,
    "quality": validate_quality,
    "case2chat": validate_case2chat,
}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in VALIDATORS:
        choices = ", ".join(sorted(VALIDATORS))
        print(f"usage: {Path(sys.argv[0]).name} <{choices}> <response-file>", file=sys.stderr)
        return 2

    validator_name, response_path = sys.argv[1:]
    try:
        text = Path(response_path).read_text(encoding="utf-8")
        documents = json_documents(text)
        VALIDATORS[validator_name](documents)
    except (OSError, UnicodeError, ResponseValidationError) as exc:
        print(f"invalid response: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
