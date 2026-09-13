import sys
from pathlib import Path


STARTER_DIR = Path(__file__).resolve().parents[1] / "starter-code"
sys.path.insert(0, str(STARTER_DIR))

from template import ChatbotBaseline


def test_mock_baseline_returns_unverified_answer_without_tools():
    user_input = "Tôi muốn xem xe VinFast"

    result = ChatbotBaseline().query(user_input)

    assert result["status"] == "success"
    assert result["mode"] == "mock_baseline"
    assert result["tool_calls"] == []
    assert "chưa được kiểm chứng" in result["answer"].lower()
    assert result["answer"] != f"[Chatbot Baseline] Trả lời cho: {user_input}"
