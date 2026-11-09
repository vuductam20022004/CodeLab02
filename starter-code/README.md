# Starter Code — Lab #4

> Dành cho học viên VinUni AI Product Labs.

## Cách Sử Dụng

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Hoàn thiện bài lab:**
   - Mở `tools.py` → Hoàn thiện 2 hàm tool + JSON Schemas.
   - Mở `template.py` → Hoàn thiện 4 TODO:
     - TODO 1: System Prompt
     - TODO 2: ChatbotBaseline
     - TODO 3: Intent Detection
     - TODO 4: Agent Loop

3. **Chạy thử:**
   ```bash
   python3 template.py
   ```

4. **Kiểm thử autograder:**
   ```bash
   python3 -m pytest ../autograder/test_agent.py -v
   ```

## Lưu Ý
- Code chạy ở chế độ **Mock Simulator** mặc định (không cần API key).
- Nếu muốn thử Live API, đặt biến môi trường: `export GEMINI_API_KEY=your_key`

## Task 3: ToolCallingAgent

Agent chạy offline theo luật nhận diện intent (keyword/regex); không gọi LLM.
`SYSTEM_PROMPT` là hợp đồng hành vi của bài lab, các quy tắc được thực thi bằng
Python. Chế độ Gemini của `ChatbotBaseline` hoạt động riêng qua `GEMINI_API_KEY`.

```python
from template import ToolCallingAgent

# Tuần tự: mỗi tool là một iteration, mặc định tối đa 5 iteration.
agent = ToolCallingAgent(max_iterations=5)
result = agent.run("Tôi muốn xem xe điện VinFast giá dưới 600 triệu.")
print(result["answer"])
print(result["trace"])

# Song song: các tool độc lập cùng chạy trong một iteration.
parallel_agent = ToolCallingAgent(max_iterations=5, parallel=True)
```

Một câu có thể yêu cầu tra cứu và ghi nhận sự cố đồng thời. Tạo ticket cần tên
(ví dụ `Tôi tên Nguyễn Văn An`) và mô tả vấn đề; nếu thiếu, agent hỏi bổ sung.
FAQ dùng một iteration và không gọi tool. Agent chỉ nêu chính sách khi có nguồn
xác nhận; không tự gán thời hạn bảo hành. Bộ phân tích theo luật có thể yêu cầu
viết lại câu khi cách diễn đạt chưa được hỗ trợ.

Kết quả gồm `answer`, `trace`, `iterations`, `status`. Trace có intent,
Thought/Action/Observation và Final Answer. Các trạng thái chính: `completed`,
`needs_clarification`, `tool_error`, `max_iterations_reached`, `out_of_scope`,
`invalid_input`. Khi hết số bước, kết quả giữ phần đã hoàn thành và nêu các tool
chưa thực hiện. Lỗi tool không được chuyển thành dữ liệu giả hoặc tự gửi lại ticket.

Hai tool đọc/ghi JSON trong `raw-data`. Ticket được ghi nguyên tử và khóa trong
cùng tiến trình; cách lưu này phù hợp cho bài lab, không thay thế cơ sở dữ liệu
cho nhiều tiến trình cùng ghi. Trace chứa thông tin khách nhập để tạo ticket;
chỉ sử dụng để kiểm tra nội bộ.

Chạy toàn bộ kiểm thử từ thư mục gốc bài lab:

```bash
python -m pytest autograder -v
```

Pytest dùng bản sao dữ liệu tạm cho từng test và tắt key môi trường trong test,
không ghi ticket vào dữ liệu gốc hoặc gọi Gemini thật. Các bài kiểm thử mới cũng
chạy độc lập bằng `python -m unittest discover -s autograder -p 'test_tool_calling_agent.py' -v`.
