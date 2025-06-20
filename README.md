# 🎥 YouTube Script & Comment Summarizer Agent

YouTube 영상의 **스크립트**와 **댓글**을 자동으로 요약해주는 LangChain 기반의 AI 에이전트입니다. 사용자는 영상 URL을 입력하고 간단한 응답만으로도 영상 요약과 댓글 반응을 빠르게 확인할 수 있습니다.

---

## 🔍 주요 기능

- 📝 **스크립트 요약**: YouTube 영상의 자막을 분석하여 핵심 내용 요약
- 💬 **댓글 요약**: 영상 댓글의 전반적인 반응을 요약
- 🧭 **라우팅 기능**: 입력된 상태(`reply`)에 따라 스크립트 또는 댓글 요약 자동 분기
- 🖥 **Streamlit UI 지원**: 간단한 웹 UI로 직접 실행 가능

---

## 🚀 사용 예시

1. **영상 URL 입력** → 스크립트 요약 자동 수행
2. **"응", "네", "보여줘"** 등의 긍정 응답 입력 → 댓글 요약 수행

### ✅ 예시 흐름

1. URL 입력:
https://youtu.be/sLe6jgHoYtk

![스크린샷 2025-06-20 160259](https://github.com/user-attachments/assets/08678cdc-d41a-408c-8b69-da22dcec4377)

2. 스크립트 요약 결과 출력

![스크린샷 2025-06-20 160328](https://github.com/user-attachments/assets/70e20c3e-2636-42a1-96d5-16387c8a704d)


3. 사용자가 "응" 또는 긍정의 표현 입력

![스크린샷 2025-06-20 160351](https://github.com/user-attachments/assets/15e2fd08-53cf-480b-b3e9-c579e911f559)

4. 댓글 요약 결과 출력

![스크린샷 2025-06-20 160405](https://github.com/user-attachments/assets/e903c56d-f38e-4cfb-8918-e2c12b966690)


출력 형태는 `script_summary` (dict) 와 `comment_summary` (string) 형태로 표시됩니다.

---

## 🗂️ 폴더 및 파일 구조

```
.
├── app2.py                 # Streamlit 기반 실행 UI
├── script_agent_05.py      # 스크립트 요약 에이전트 정의
├── comment_agent_05.py     # 댓글 요약 에이전트 정의
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 설명 파일
```

---

## ⚙️ 설정 방법

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

루트 디렉토리에 `.env` 파일 생성 후 다음 내용을 입력:

```env
OPENAI_API_KEY=your_openai_key_here
```

---

## ▶️ 실행 방법

```bash
streamlit run app2.py
```

브라우저에서 실행된 페이지로 이동하여 URL 입력 후 요약 기능을 사용하세요.

---

## 🧠 설정 옵션

```python
config = {"configurable": {"thread_id": "your_thread_id"}}
```

* `thread_id`는 세션별 상태를 구분하는 키입니다.
* Streamlit에서는 `st.session_state` + `uuid.uuid4()` 조합으로 고유 세션 ID를 생성해 사용 가능합니다.
* 이 설정은 `MemorySaver`와 함께 LangGraph 내부 상태를 세션 단위로 안전하게 저장/관리하는 데 사용됩니다.

---

## 🔩 핵심 기술 스택

* **Python**
* **LangChain**, **LangGraph**
* **Streamlit** (웹 UI 프레임워크)
* **OpenAI GPT-4o-mini**
* **typing\_extensions**, **dotenv**, **pydantic**
