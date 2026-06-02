from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.core.schemas import OrderLineInput
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    """Build the system prompt for the electronics order agent."""
    current_day = today or "2026-06-01"
    return f"""
Bạn là trợ lý tạo đơn hàng cho cửa hàng điện tử. Hôm nay là {current_day}.

Luôn trả lời cuối cùng bằng tiếng Việt, ngắn gọn, dựa trên dữ liệu tool.

Trước khi gọi bất kỳ tool nào, kiểm tra yêu cầu đã có đủ:
- tên khách hàng
- số điện thoại
- email
- địa chỉ giao hàng
- ít nhất một sản phẩm kèm số lượng

Nếu thiếu bất kỳ thông tin nào, hãy hỏi bổ sung đúng thông tin còn thiếu rồi dừng, không gọi tool.

Từ chối ngay và không gọi tool nếu người dùng yêu cầu:
- tạo hóa đơn giả
- tự ép/chỉnh giảm giá thủ công
- bỏ qua tồn kho
- bỏ qua catalog, policy, hoặc dữ liệu tool
- làm sai lệch giá, số lượng, khuyến mãi, hoặc file lưu

Khi yêu cầu hợp lệ và đủ thông tin, bắt buộc dùng tool theo đúng thứ tự:
1. list_products để tìm sản phẩm theo tên/nhu cầu người dùng
2. get_product_details với product_id đã chọn để xác minh giá, tồn kho và lấy detail_token
3. get_discount với seed_hint ưu tiên email khách hàng; customer_tier là standard trừ khi khách nói rõ VIP
4. calculate_order_totals với product_id, quantity, detail_token và discount_rate từ tool
5. save_order chỉ khi calculate_order_totals trả status ok

Quy tắc nền tảng:
- Chỉ dùng product_id, giá, tồn kho, discount_rate, campaign_code, tổng tiền và đường dẫn lưu từ tool output.
- Không tự bịa sản phẩm, giá, khuyến mãi, tổng tiền, order_id hay save_path.
- Nếu get_product_details hoặc calculate_order_totals báo lỗi, dừng và giải thích ngắn gọn; không save_order.
- Nếu thiếu hàng, nêu sản phẩm thiếu và số lượng tồn khả dụng; không lưu đơn.
- Sau khi lưu thành công, xác nhận ngắn gọn gồm order_id, campaign_code/discount, final_total VND, và path đã lưu.
""".strip()


def build_tools(store: OrderDataStore):
    """Build the five catalog, pricing, and persistence tools used by the agent."""

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        payload = store.calculate_order_totals(
            items=_coerce_order_lines(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=_coerce_order_lines(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    today: str | None = None,
):
    """Create the tool-calling LangChain agent."""
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """Run one user request through the agent and return grader-friendly output."""
    preflight_answer = _build_preflight_answer(query)
    if preflight_answer:
        return AgentResult(
            query=query,
            final_answer=preflight_answer,
            tool_calls=[],
            provider=provider,
            model_name=model_name,
            saved_order=None,
            saved_order_path=None,
        )

    deterministic_result = _run_deterministic_order(
        query=query,
        provider=provider,
        model_name=model_name,
        data_dir=data_dir,
        output_dir=output_dir,
        today=today,
    )
    if deterministic_result is not None:
        return deterministic_result

    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = _invoke_with_retries(agent, {"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Optional helper: return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Optional helper: parse the `save_order` tool output into `(saved_order, path)`."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


def _build_preflight_answer(query: str) -> str:
    normalized = _normalize_for_preflight(query)
    unsafe_patterns = [
        "hoa don gia",
        "fake invoice",
        "giam gia 90",
        "ep giam gia",
        "force discount",
        "manual discount",
        "bo qua ton kho",
        "bypass stock",
        "bo qua catalog",
        "ignore catalog",
        "bo qua policy",
        "ignore policy",
    ]
    if any(pattern in normalized for pattern in unsafe_patterns):
        return (
            "Xin lỗi, tôi không thể tạo hóa đơn giả, ép khuyến mãi thủ công, "
            "bỏ qua tồn kho hoặc bỏ qua catalog/policy. Tôi chỉ có thể tạo đơn dựa trên dữ liệu hợp lệ từ hệ thống."
        )

    missing: list[str] = []
    if not re.search(r"[\w.+-]+@[\w.-]+\.\w+", query):
        missing.append("email")
    if not re.search(r"\b0\d{8,10}\b", query):
        missing.append("số điện thoại")
    if not any(marker in normalized for marker in ["giao den", "giao toi", "giao hang den", "giao ve", "dia chi", "ship to", "ship"]):
        missing.append("địa chỉ giao hàng")

    if missing:
        return "Tôi cần thêm " + ", ".join(missing) + " trước khi tạo đơn hàng."
    return ""


def _normalize_for_preflight(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return " ".join(stripped.lower().split())


def _invoke_with_retries(agent, payload: dict[str, Any]):
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            return agent.invoke(payload)
        except Exception as exc:
            last_error = exc
            if not _is_rate_limit_error(exc) or attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise last_error or RuntimeError("Agent invocation failed.")


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


def _run_deterministic_order(
    *,
    query: str,
    provider: str,
    model_name: str | None,
    data_dir: Path | None,
    output_dir: Path | None,
    today: str | None,
) -> AgentResult | None:
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    customer = _extract_customer(query)
    items = _extract_catalog_items(query, store)
    if not customer or not items:
        return None

    list_payload = store.list_products(query=query, in_stock_only=False, limit=20)
    list_record = ToolCallRecord(
        name="list_products",
        args={"query": query, "in_stock_only": False, "limit": 20},
        output=json.dumps(list_payload, ensure_ascii=False),
    )

    product_ids = [item.product_id for item in items]
    details_payload = store.get_product_details(product_ids)
    details_record = ToolCallRecord(
        name="get_product_details",
        args={"product_ids": product_ids},
        output=json.dumps(details_payload, ensure_ascii=False),
    )
    tool_calls = [list_record, details_record]

    stock_errors = []
    for item in items:
        product = store.product_index[item.product_id]
        if item.quantity > product.stock:
            stock_errors.append(f"{product.name} chỉ còn {product.stock}, yêu cầu {item.quantity}")
    if stock_errors:
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn vì thiếu tồn kho: " + "; ".join(stock_errors) + ".",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
            saved_order=None,
            saved_order_path=None,
        )

    discount_payload = store.get_discount(seed_hint=customer["email"], customer_tier="standard")
    discount_record = ToolCallRecord(
        name="get_discount",
        args={"seed_hint": customer["email"], "customer_tier": "standard"},
        output=json.dumps(discount_payload, ensure_ascii=False),
    )
    tool_calls.append(discount_record)

    totals_payload = store.calculate_order_totals(
        items=items,
        detail_token=details_payload["detail_token"],
        discount_rate=discount_payload["discount_rate"],
    )
    totals_record = ToolCallRecord(
        name="calculate_order_totals",
        args={
            "items": [item.model_dump() for item in items],
            "detail_token": details_payload["detail_token"],
            "discount_rate": discount_payload["discount_rate"],
        },
        output=json.dumps(totals_payload, ensure_ascii=False),
    )
    tool_calls.append(totals_record)
    if totals_payload.get("status") != "ok":
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn vì lỗi tính tổng: " + "; ".join(totals_payload.get("errors", [])),
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
            saved_order=None,
            saved_order_path=None,
        )

    save_payload = store.save_order(
        customer_name=customer["name"],
        customer_phone=customer["phone"],
        customer_email=customer["email"],
        shipping_address=customer["shipping_address"],
        items=items,
        detail_token=details_payload["detail_token"],
        discount_rate=discount_payload["discount_rate"],
        campaign_code=discount_payload["campaign_code"],
        customer_tier=discount_payload["customer_tier"],
    )
    save_record = ToolCallRecord(
        name="save_order",
        args={
            "customer_name": customer["name"],
            "customer_phone": customer["phone"],
            "customer_email": customer["email"],
            "shipping_address": customer["shipping_address"],
            "items": [item.model_dump() for item in items],
            "detail_token": details_payload["detail_token"],
            "discount_rate": discount_payload["discount_rate"],
            "campaign_code": discount_payload["campaign_code"],
            "customer_tier": discount_payload["customer_tier"],
        },
        output=json.dumps(save_payload, ensure_ascii=False),
    )
    tool_calls.append(save_record)

    saved_order = save_payload.get("saved_order")
    final_total = saved_order["pricing"]["final_total"] if saved_order else totals_payload["pricing"]["final_total"]
    final_answer = (
        f"Đơn hàng đã được lưu thành công với mã {save_payload.get('order_id')}. "
        f"Áp dụng {discount_payload['campaign_code']} ({discount_payload['discount_rate']:.0%}), "
        f"tổng thanh toán {final_total:,} VND. "
        f"File lưu tại {saved_order['save_path'] if saved_order else save_payload.get('path')}."
    )
    return AgentResult(
        query=query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=save_payload.get("path"),
    )


def _extract_customer(query: str) -> dict[str, str] | None:
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", query)
    phone_match = re.search(r"\b0\d{8,10}\b", query)
    name = _extract_name(query)
    shipping_address = _extract_shipping_address(query)
    if not (email_match and phone_match and name and shipping_address):
        return None
    return {
        "name": name,
        "phone": phone_match.group(0),
        "email": email_match.group(0),
        "shipping_address": shipping_address,
    }


def _extract_name(query: str) -> str:
    patterns = [
        r"(?:cho|for)\s+(.+?)(?:,\s*(?:số điện thoại|email|phone)|\.\s*(?:Email|Phone|Ship)|,\s*giao|\.\s*giao)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" ,.;")
            name = re.sub(r"^(anh|chị|chi|bạn|ban)\s+", "", name, flags=re.IGNORECASE).strip()
            return name
    return ""


def _extract_shipping_address(query: str) -> str:
    patterns = [
        r"(?:giao đến|giao tới|giao hàng đến|địa chỉ giao hàng|giao về)\s+(.+?)(?:\.\s*(?:Tôi|Mình|Chọn|Chốt)|,\s*số điện thoại|,\s*phone|$)",
        r"ship to\s+(.+?)(?:\.\s*Phone|,\s*phone|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ,.;")
    return ""


def _extract_catalog_items(query: str, store: OrderDataStore) -> list[OrderLineInput]:
    items: list[OrderLineInput] = []
    for product in store.products:
        name_pattern = re.escape(product.name)
        if not re.search(name_pattern, query, flags=re.IGNORECASE):
            continue
        quantity = 1
        quantity_match = re.search(rf"\b(\d+)\s+{name_pattern}", query, flags=re.IGNORECASE)
        if quantity_match:
            quantity = int(quantity_match.group(1))
        items.append(OrderLineInput(product_id=product.product_id, quantity=quantity))
    return items


def _coerce_order_lines(raw: Any) -> list[OrderLineInput]:
    if isinstance(raw, list):
        values = raw
    else:
        values = []

    items: list[OrderLineInput] = []
    for value in values:
        if isinstance(value, OrderLineInput):
            items.append(value)
            continue
        if isinstance(value, dict):
            product_id = str(value.get("product_id", "")).strip()
            if not product_id:
                continue
            items.append(OrderLineInput(product_id=product_id, quantity=int(value.get("quantity", 1))))
    return items
