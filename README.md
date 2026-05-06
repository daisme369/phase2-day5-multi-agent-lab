# Lab 20: Multi-Agent Research System

Repo này cung cấp **production-grade skeleton** cho hệ thống **Multi-Agent** với chức năng nghiên cứu (Research) được xây dựng dựa trên LangGraph. Hệ thống được thiết kế để phân phối luồng công việc giữa các agent chuyên trách nhằm tự động hóa quy trình thu thập, phân tích và tổng hợp thông tin.

---

## 1. Multi Agent System Architecture

Hệ thống hoạt động theo mô hình Hierarchical / Router-Worker, bao gồm một **Supervisor Agent** (đóng vai trò điều phối) và các **Worker Agents** (thực hiện công việc chuyên môn).

### Sơ đồ kiến trúc (Architecture Diagram)

```text
User Query
   │
   ▼
[ Supervisor / Router Agent ] ─── Phân loại Query & Lên kế hoạch
   │
   ├── Xử lý nghiên cứu
   │   └──► [ Researcher Agent ] ──► (Tìm kiếm, thu thập dữ liệu) ──► `research_notes`
   │
   ├── Phân tích dữ liệu
   │   └──► [ Analyst Agent ] ──► (Xử lý, trích xuất insight) ──► `analysis_notes`
   │
   └── Viết báo cáo
       └──► [ Writer Agent ] ──► (Tổng hợp, định dạng markdown) ──► `final_answer`
   │
   ▼
[ Trace + Benchmark Report ] ─── (Ghi nhận log, đánh giá hiệu năng)
```

### Chức năng của từng thành phần:

- **Supervisor (Router):** Nhận câu hỏi từ user, đánh giá độ phức tạp, và điều hướng các task đến đúng Agent. Supervisor duy trì state chung (`State`) của toàn hệ thống trong LangGraph.
- **Researcher:** Nhận nhiệm vụ từ Supervisor, sử dụng công cụ tìm kiếm (ví dụ: Tavily API) hoặc mock data để trích xuất thông tin mới nhất từ internet/database.
- **Analyst:** Xử lý dữ liệu thô do Researcher trả về, lọc bỏ thông tin nhiễu, đánh giá độ tin cậy và tìm ra các insight cốt lõi.
- **Writer:** Dựa vào insight đã phân tích để viết báo cáo tổng hợp hoàn chỉnh, đúng định dạng và gửi lại kết quả cuối cùng cho người dùng.

Tất cả các Agents giao tiếp với nhau thông qua **Shared State** của LangGraph. Hệ thống cũng tích hợp các production guardrails thiết yếu như: giới hạn vòng lặp (`max_iterations`), xử lý timeout, retry/fallback, và data validation bằng Pydantic.

---

## 2. Hướng dẫn Step-by-Step Set Up và Sử dụng hệ thống

### Bước 1: Clone và Cài đặt môi trường

Yêu cầu: Python 3.10+ đã được cài đặt sẵn trên máy tính.

```bash
# 1. Clone repository (nếu bạn chưa clone)
# git clone <your-repo-url>
# cd phase2-day5-multi-agent-lab

# 2. Tạo virtual environment
python -m venv .venv

# 3. Kích hoạt virtual environment
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
# source .venv/bin/activate

# 4. Cài đặt các thư viện phụ thuộc và tools cho dev
pip install -e ".[dev]"
```

### Bước 2: Cấu hình biến môi trường (Environment Variables)

Hệ thống yêu cầu các API keys để gọi LLM và sử dụng công cụ tìm kiếm.

```bash
# Copy template file cấu hình môi trường
cp .env.example .env
```

Mở file `.env` bằng code editor và điền các thông tin cần thiết:

```env
# (Bắt buộc) API Key cho LLM
OPENAI_API_KEY=your_openai_api_key_here

# (Tuỳ chọn) Tracing & Observability
LANGSMITH_API_KEY=your_langsmith_api_key_here

# (Tuỳ chọn) Web Search API
TAVILY_API_KEY=your_tavily_api_key_here
```

### Bước 3: Kiểm tra hệ thống (Smoke Test)

Xác minh cấu trúc file và CLI hoạt động trơn tru.

```bash
# Chạy bộ unit tests mặc định
make test

# Hiển thị menu hướng dẫn của Command Line Interface (CLI)
python -m multi_agent_research_lab.cli --help
```

### Bước 4: Chạy thử Baseline (Single-Agent)

Lệnh dưới đây sẽ gọi phiên bản chạy bằng 1 Agent duy nhất. Baseline này được dùng làm tham chiếu (benchmark) so sánh với hệ thống nhiều Agent.

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

### Bước 5: Khởi chạy hệ thống Multi-Agent

Kích hoạt workflow hoàn chỉnh để quan sát sự tương tác giữa Supervisor, Researcher, Analyst và Writer:

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

*(Lưu ý: Mặc định trong starter template, một số luồng LLM client và logic của Agent đang được để trống (TODO). Hệ thống sẽ nhắc nhở bạn implement các phần logic trong thư mục `src/multi_agent_research_lab/` để hệ thống thực sự tạo ra được output cuối cùng).*

### Bước 6: Giám sát và Đánh giá (Observability & Benchmark)

- **Tracing**: Truy cập tài khoản [LangSmith](https://smith.langchain.com/) (nếu đã cấu hình `LANGSMITH_API_KEY`) để xem trace luồng chạy, giúp bạn giải thích chính xác agent nào đã làm gì.
- **Benchmark**: Dùng tính năng evaluation để sinh báo cáo so sánh kết quả của Bước 4 (Single-agent) vs Bước 5 (Multi-agent) về mặt chất lượng, độ trễ (latency), và chi phí (cost). Báo cáo sẽ được xuất vào thư mục `reports/`.

---

## References

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI Agents SDK orchestration/handoffs](https://developers.openai.com/api/docs/guides/agents/orchestration)
- [LangGraph concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [LangSmith tracing](https://docs.smith.langchain.com/)
- [Langfuse tracing](https://langfuse.com/docs)
