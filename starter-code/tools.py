import json
import os
import tempfile
import threading
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")
_TICKET_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not isinstance(category, str) or category.lower() not in ("xe_dien", "du_lich"):
        return [{"error": "Danh mục không hợp lệ."}]
    if type(max_price) is not int or max_price < 0:
        return [{"error": "Giá tối đa phải là số nguyên không âm (VNĐ)."}]
    try:
        with open(catalog_file, encoding="utf-8") as source:
            products = json.load(source)
        if not isinstance(products, list) or any(
            not isinstance(p, dict)
            or not isinstance(p.get("name"), str)
            or not isinstance(p.get("category"), str)
            or type(p.get("price_vnd")) is not int
            or p["price_vnd"] < 0
            for p in products
        ):
            return [{"error": "Dữ liệu danh mục không hợp lệ."}]
        return [p for p in products if p["category"].lower() == category.lower()
                and p["price_vnd"] <= max_price]
    except (OSError, ValueError):
        return [{"error": "Không thể đọc danh mục sản phẩm."}]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    if (not isinstance(customer_name, str) or not customer_name.strip()
            or not isinstance(issue_description, str) or not issue_description.strip()):
        return {"error": "Cần tên khách hàng và mô tả vấn đề."}
    if not isinstance(priority, str) or priority.lower() not in ("low", "medium", "high"):
        return {"error": "Mức ưu tiên không hợp lệ."}
    temp_path = None
    try:
        # Hai lượt tạo ticket trong cùng tiến trình không được ghi đè nhau.
        with _TICKET_LOCK:
            tickets = []
            if os.path.exists(tickets_file):
                with open(tickets_file, encoding="utf-8") as source:
                    tickets = json.load(source)
            if not isinstance(tickets, list) or any(
                not isinstance(t, dict) or not isinstance(t.get("ticket_id"), str)
                for t in tickets
            ):
                return {"error": "Dữ liệu ticket không hợp lệ; chưa ghi yêu cầu mới."}
            now = datetime.now(timezone(timedelta(hours=7)))
            prefix = "TK-" + now.strftime("%Y%m%d") + "-"
            used_ids = {t["ticket_id"] for t in tickets}
            sequence = 1
            while f"{prefix}{sequence:03d}" in used_ids:
                sequence += 1
            ticket = {
                "ticket_id": f"{prefix}{sequence:03d}",
                "customer_name": customer_name.strip(),
                "issue_description": issue_description.strip(),
                "priority": priority.lower(),
                "status": "open",
                "created_at": now.isoformat(),
                "category": "general"
            }
            tickets.append(ticket)
            # Ghi file tạm rồi thay thế để tránh để lại JSON ghi dở khi có lỗi.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=RAW_DATA_DIR,
                prefix=".tickets-", suffix=".json", delete=False
            ) as target:
                temp_path = target.name
                json.dump(tickets, target, indent=2, ensure_ascii=False)
                target.write("\n")
            os.replace(temp_path, tickets_file)
            temp_path = None
            return dict(ticket, message="Đã ghi nhận yêu cầu hỗ trợ.")
    except (OSError, ValueError):
        return {"error": "Không thể lưu ticket; chưa xác nhận tạo yêu cầu thành công."}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu danh mục sản phẩm Vingroup theo loại và giá tối đa VNĐ.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["xe_dien", "du_lich"]},
                "max_price": {"type": "integer", "minimum": 0}
            },
            "required": ["category"],
            "additionalProperties": False
        }
    },
    {
        "name": "submit_support_ticket",
        "description": "Ghi nhận yêu cầu hỗ trợ và trả về mã ticket thực tế.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "minLength": 1},
                "issue_description": {"type": "string", "minLength": 1},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]}
            },
            "required": ["customer_name", "issue_description"],
            "additionalProperties": False
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
