from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DemoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.assets = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"])
        if tag == "link" and values.get("rel") == "stylesheet":
            self.assets.append(values["href"])


def test_static_demo_contract_and_assets_exist():
    parser = DemoParser()
    parser.feed((ROOT / "index.html").read_text())
    assert {
        "bodyCanvas",
        "fxCanvas",
        "pulseCanvas",
        "woundButton",
        "runButton",
        "healthValue",
        "activeScore",
        "randomScore",
    } <= parser.ids
    for asset in parser.assets:
        assert (ROOT / asset).is_file(), asset
    assert (ROOT / "assets/pulse-repair-card.png").is_file()


def test_browser_code_names_the_scalar_boundary_and_attackers():
    source = (ROOT / "web/app.js").read_text()
    assert "function measuredPulse(candidate)" in source
    assert "async function adaptiveSearch(input, budget, repairSlots, token)" in source
    assert "function selectRandom(input, budget, repairSlots, seed)" in source
    assert "function selectExhaustive(input, repairSlots)" in source


def test_live_demo_links_to_reusable_pulse_triage_tool():
    html = (ROOT / "index.html").read_text()
    assert "docs/PULSE_TRIAGE.md" in html
    assert "USE PULSETRIAGE" in html
