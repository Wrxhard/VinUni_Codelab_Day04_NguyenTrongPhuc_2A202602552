"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Triển khai chatbot baseline và agent có khả năng gọi công cụ.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE 1: SYSTEM PROMPT cấp sản xuất
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
## PERSONA
Bạn là **VinAssistant** — trợ lý AI chuyên nghiệp của Vingroup.
- **Vai trò:** Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl, v.v.)
- **Giọng nói:** Chuyên nghiệp, thân thiện.
- **Trách nhiệm:** Giải đáp câu hỏi khách hàng bằng dữ liệu thực tế

## AVAILABLE TOOLS
Bạn có quyền truy cập 2 công cụ sau:

1. **search_product_catalog**
   - Mô tả: Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục (xe_dien, du_lich) và giá
   - Input: category (string), max_price (int, tùy chọn)
   - Output: Danh sách sản phẩm chi tiết (tên, giá, mô tả)
   - Khi dùng: Khách hàng hỏi về sản phẩm, giá cả, hoặc so sánh

2. **submit_support_ticket**
   - Mô tả: Ghi nhận vấn đề hỗ trợ khách hàng vào hệ thống ticket
   - Input: customer_name, issue_description, priority (low/medium/high)
   - Output: ticket_id, status, created_at
   - Khi dùng: Khách hàng yêu cầu hỗ trợ, báo cáo lỗi, hoặc có khiếu nại

## CORE RULES
1. **KHÔNG BAO GIỜ bịa dữ liệu:** Nếu bạn không chắc chắn về sản phẩm, giá cả, hoặc dịch vụ,
   BẮT BUỘC gọi tool search_product_catalog để lấy dữ liệu thực.

2. **PHẢI gọi tool khi cần:** Nếu câu hỏi liên quan đến:
   - Thông tin sản phẩm Vingroup → search_product_catalog
   - Yêu cầu hỗ trợ từ khách hàng → submit_support_ticket

3. **Không giả vờ hiểu:** Nếu câu hỏi nằm ngoài kiến thức của bạn hoặc không có tool phù hợp,
   hãy thành thật nói rằng bạn cần hỗ trợ thêm.

4. **Cung cấp bối cảnh:** Luôn giải thích tại sao bạn chọn tool nào và đang tìm gì.

## OPERATIONAL BOUNDARIES
- **Phạm vi:** Chỉ trả lời câu hỏi liên quan đến sản phẩm & dịch vụ Vingroup
- **Out-of-scope:** Chủ đề không liên quan đến Vingroup
- **Hành động:** Nếu câu hỏi ngoài phạm vi, từ chối lịch sự

## OUTPUT CONTRACT
Tuân thủ format ReAct sau:

**Thought:** Phân tích câu hỏi, xác định cần gọi tool nào (nếu có).
**Action:** Gọi tool nếu cần, kèm theo tham số cụ thể.
**Observation:** Kết quả trả về từ tool (hoặc "No tool needed").
**Final Answer:** Trả lời khách hàng một cách rõ ràng, ngắn gọn, có sử dụng dữ liệu từ tool.

---

**Lưu ý:** Hãy suy nghĩ cẩn thận trước khi trả lời. Luôn cung cấp thông tin chính xác và đáng tin cậy.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": (
                f"[Chatbot Baseline] Thông tin trả lời cho “{user_input}” "
                "chưa được kiểm chứng vì chatbot không sử dụng công cụ tra cứu."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        normalized_input = " ".join(user_input.casefold().split())

        catalog_keywords = (
            "giá", "gia", "mua", "xem", "tìm", "tim", "sản phẩm", "san pham",
            "xe điện", "xe dien", "vinfast", "vinpearl", "du lịch", "du lich",
        )
        ticket_keywords = (
            "hỗ trợ", "ho tro", "báo lỗi", "bao loi", "lỗi", "loi", "hỏng",
            "hong", "sự cố", "su co", "khiếu nại", "khieu nai", "ticket",
        )

        needs_ticket = any(keyword in normalized_input for keyword in ticket_keywords)
        faq_keywords = ("bảo hành", "bao hanh", "chính sách", "chinh sach", "bao lâu", "bao lau")
        explicit_catalog_keywords = (
            "giá", "gia", "mua", "xem", "tìm", "tim", "sản phẩm", "san pham",
            "du lịch", "du lich", "vinpearl",
        )
        has_catalog_signal = any(keyword in normalized_input for keyword in catalog_keywords)
        has_explicit_catalog_signal = any(
            keyword in normalized_input for keyword in explicit_catalog_keywords
        )
        is_faq = (
            any(keyword in normalized_input for keyword in faq_keywords)
            and not has_explicit_catalog_signal
            and not needs_ticket
        )
        needs_catalog = (
            has_catalog_signal
            and not is_faq
            and (not needs_ticket or has_explicit_catalog_signal)
        )
        intents = {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq or (not needs_catalog and not needs_ticket),
        }

        self.trace.append({"step": "init", "user_input": user_input, "intents": intents})

        if intents["is_faq"]:
            answer = (
                "Về chính sách bảo hành, thời hạn và điều kiện áp dụng phụ thuộc vào "
                "từng sản phẩm. Vui lòng cung cấp tên hoặc phiên bản sản phẩm cụ thể "
                "để tôi hỗ trợ chính xác hơn."
            )
            self.trace.append({"step": "final_answer", "answer": answer})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed",
            }

        actions = []
        if intents["needs_catalog"]:
            actions.append("search_product_catalog")
        if intents["needs_ticket"]:
            actions.append("submit_support_ticket")

        iteration = 0
        observations = []
        while actions and iteration < self.max_iterations:
            tool_name = actions.pop(0)
            iteration += 1

            if tool_name == "search_product_catalog":
                category = (
                    "du_lich"
                    if any(keyword in normalized_input for keyword in ("du lịch", "du lich", "vinpearl"))
                    else "xe_dien"
                )
                arguments = {"category": category}
                price_match = re.search(
                    r"(\d+(?:[.,]\d+)?)\s*(tỷ|tỉ|ty|ti|triệu|trieu|nghìn|nghin)",
                    normalized_input,
                )
                if price_match:
                    multipliers = {
                        "tỷ": 1_000_000_000,
                        "tỉ": 1_000_000_000,
                        "ty": 1_000_000_000,
                        "ti": 1_000_000_000,
                        "triệu": 1_000_000,
                        "trieu": 1_000_000,
                        "nghìn": 1_000,
                        "nghin": 1_000,
                    }
                    amount = float(price_match.group(1).replace(",", "."))
                    arguments["max_price"] = int(amount * multipliers[price_match.group(2)])
            else:
                name_match = re.search(
                    r"(?:tôi tên|toi ten|tên tôi là|ten toi la)\s+([^,.]+)",
                    normalized_input,
                )
                customer_name = name_match.group(1).strip().title() if name_match else "Khách hàng"
                high_priority_keywords = (
                    "nghiêm trọng", "nghiem trong", "khẩn cấp", "khan cap", "gấp", "gap",
                )
                priority = (
                    "high"
                    if any(keyword in normalized_input for keyword in high_priority_keywords)
                    else "medium"
                )
                arguments = {
                    "customer_name": customer_name,
                    "issue_description": user_input.strip(),
                    "priority": priority,
                }

            self.trace.append(
                {"step": "action", "iteration": iteration, "tool": tool_name, "arguments": arguments}
            )
            try:
                observation = TOOL_MAP[tool_name](**arguments)
            except Exception as exc:
                error = f"Không thể thực thi {tool_name}: {exc}"
                self.trace.append(
                    {"step": "observation", "iteration": iteration, "tool": tool_name, "error": str(exc)}
                )
                return {
                    "answer": error,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "error",
                }

            observations.append((tool_name, arguments, observation))
            self.trace.append(
                {
                    "step": "observation",
                    "iteration": iteration,
                    "tool": tool_name,
                    "result": observation,
                }
            )

        if actions:
            answer = "Lỗi: Vượt quá số bước tối đa trước khi hoàn thành yêu cầu."
            self.trace.append({"step": "final_answer", "answer": answer})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "max_iterations_reached",
            }

        answer_parts = []
        for tool_name, arguments, observation in observations:
            if tool_name == "search_product_catalog":
                if observation and "error" in observation[0]:
                    answer = f"Không thể tra cứu danh mục sản phẩm: {observation[0]['error']}"
                    self.trace.append({"step": "final_answer", "answer": answer})
                    return {
                        "answer": answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "error",
                    }
                if not observation:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    products = [
                        f"- {product['name']}: {product['price_vnd']:,} VNĐ. "
                        f"{product.get('description', '')}".rstrip()
                        for product in observation
                    ]
                    answer_parts.append("Các sản phẩm phù hợp:\n" + "\n".join(products))
            else:
                ticket_id = observation.get("ticket_id", "không xác định")
                status = observation.get("status", "unknown")
                answer_parts.append(
                    f"Đã tạo phiếu hỗ trợ {ticket_id} cho {arguments['customer_name']} "
                    f"(trạng thái: {status})."
                )

        answer = "\n\n".join(answer_parts)
        self.trace.append({"step": "final_answer", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
