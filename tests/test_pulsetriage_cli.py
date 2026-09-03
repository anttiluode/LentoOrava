import json
from pathlib import Path
import sys

from pulsetriage.cli import _score_from_stdout, main


def test_score_parser_accepts_number_and_json_object():
    assert _score_from_stdout("log line\n1.25\n") == 1.25
    assert _score_from_stdout('{"score": 2.5}\n') == 2.5


def test_cli_passes_rollback_set_through_environment(tmp_path: Path):
    items = [f"part-{index}" for index in range(64)]
    item_file = tmp_path / "items.txt"
    item_file.write_text("\n".join(items) + "\n", encoding="utf-8")
    output = tmp_path / "result.json"
    evaluator = (
        "import json,os;"
        "r=set(json.loads(os.environ['PULSE_TRIAGE_ROLLBACK_JSON']));"
        "h={'part-11':4.0,'part-52':2.0};"
        "print(100-sum(h.get(x,0) for x in r))"
    )

    return_code = main(
        [
            "--items",
            str(item_file),
            "--budget",
            "29",
            "--max-suspects",
            "2",
            "--screening-pairs",
            "12",
            "--shortlist-size",
            "4",
            "--seed",
            "2",
            "--lower-is-better",
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            evaluator,
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert return_code == 0
    assert [suspect["item"] for suspect in payload["suspects"]] == [
        "part-11",
        "part-52",
    ]
