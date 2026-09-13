import json
import sys
from pathlib import Path


STARTER_DIR = Path(__file__).resolve().parents[1] / "starter-code"
sys.path.insert(0, str(STARTER_DIR))

import tools
from template import ToolCallingAgent


def test_tool_definitions_expose_both_tools_with_required_parameters():
    definitions = {definition["name"]: definition for definition in tools.TOOL_DEFINITIONS}

    assert set(definitions) == {"search_product_catalog", "submit_support_ticket"}
    assert definitions["search_product_catalog"]["parameters"]["required"] == ["category"]
    assert definitions["submit_support_ticket"]["parameters"]["required"] == [
        "customer_name",
        "issue_description",
    ]


def test_submit_support_ticket_appends_without_overwriting_existing_ticket(tmp_path, monkeypatch):
    tickets_file = tmp_path / "support_tickets.json"
    existing_ticket = {"ticket_id": "TK-20240101-001", "status": "closed"}
    tickets_file.write_text(json.dumps([existing_ticket]), encoding="utf-8")
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))

    result = tools.submit_support_ticket("Lan", "Xe không khởi động", "HIGH")
    saved_tickets = json.loads(tickets_file.read_text(encoding="utf-8"))

    assert saved_tickets[0] == existing_ticket
    assert saved_tickets[1]["ticket_id"].endswith("-002")
    assert saved_tickets[1]["priority"] == "high"
    assert result["ticket_id"] == saved_tickets[1]["ticket_id"]
    assert result["status"] == "open"


def test_catalog_reports_missing_data_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))

    assert tools.search_product_catalog("xe_dien") == [
        {"error": "Product catalog file not found."}
    ]


def test_agent_stops_when_no_iteration_is_available():
    result = ToolCallingAgent(max_iterations=0).run("Tìm xe điện VinFast dưới 600 triệu")

    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 0
    assert "vượt quá" in result["answer"].lower()


def test_agent_runs_catalog_and_ticket_for_a_combined_request(tmp_path, monkeypatch):
    source_catalog = Path(__file__).resolve().parents[1] / "raw-data" / "product_catalog.json"
    (tmp_path / "product_catalog.json").write_text(
        source_catalog.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "support_tickets.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))

    result = ToolCallingAgent(max_iterations=5).run(
        "Tôi tên Lan, muốn xem xe điện dưới 600 triệu và cần hỗ trợ lỗi sạc."
    )

    assert result["status"] == "completed"
    assert result["iterations"] == 2
    assert "VF 3" in result["answer"]
    assert "TK-" in result["answer"]


def test_agent_reports_catalog_file_error_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))

    result = ToolCallingAgent().run("Tìm xe điện VinFast dưới 600 triệu")

    assert result["status"] == "error"
    assert "không thể tra cứu" in result["answer"].lower()
