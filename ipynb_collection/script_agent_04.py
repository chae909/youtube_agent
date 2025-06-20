# %%
from urllib.parse import urlparse
from langgraph.graph import MessagesState, StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import json

# %%
load_dotenv()

# %%
llm = ChatOpenAI(model="gpt-4o", streaming=True)

# %%
class AgentState(MessagesState):
    pass

graph_builder = StateGraph(AgentState)

# %%
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    import re
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    
    parsed_url = urlparse(url)
    if parsed_url.hostname == "googleusercontent.com" and parsed_url.path.startswith('/youtube.com/'):
        return parsed_url.path.split('/')[-1]
    
    return None

# %%
def get_youtube_transcript(state: AgentState) -> dict:
    """
    [수정됨] 스크립트 또는 에러 메시지를 AIMessage에 담아 messages 리스트에 추가합니다.
    """
    print("🚀 [Tool] get_youtube_transcript 호출됨")
    user_url = state["messages"][-1].content
    
    try:
        video_id = extract_video_id(user_url)
        if not video_id:
            raise ValueError("유효한 유튜브 URL에서 Video ID를 추출할 수 없습니다.")

        print(f"✅ 영상 ID 추출 성공: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        transcript_text = " ".join([item['text'] for item in transcript_list])

        if len(transcript_text) < 100:
            raise ValueError("자막 내용이 너무 짧아 요약할 수 없습니다.")
        
        if len(transcript_text) > 15000:
            print(f"⚠️ 자막 크기 초과. 일부만 사용 (최대 15000자)")
            transcript_text = transcript_text[:15000]

        print("✅ 1. 자막 추출 성공")
        # 성공 시, 자막 내용을 메시지로 반환
        return {"messages": [AIMessage(content=transcript_text)]}

    except Exception as e:
        error_message = f"ERROR: 자막 추출 중 오류 발생 - {e}"
        print(f"🚨 {error_message}")
        # 실패 시, 에러 내용을 메시지로 반환
        return {"messages": [AIMessage(content=error_message)]}

# %%
summarize_prompt = (
    "너는 유튜브 영상의 스크립트(자막)를 분석해, 간결하고 보기 좋은 **JSON 형식 요약**을 작성하는 AI 전문가입니다.\n\n"

    "🔹 **목표**\n"
    "스크립트를 바탕으로 다음 항목들을 포함한 JSON 객체를 생성하세요. 각 항목은 정확하고 친절한 말투로 작성하되, 너무 길지 않게 요약하세요.\n\n"

    "🔹 **출력 형식(JSON)**\n"
    "{\n"
    '  "요약": "영상의 핵심 내용을 줄바꿈 포함하여 부드럽게 설명",\n'
    '  "운동 강도": "예: 초급자용, 모든 레벨, 고강도 등",\n'
    '  "운동 루틴": [\n'
    '    "1. 🧘‍♀️ 동작 이름 - 간단한 설명",\n'
    '    "2. 🤲 동작 이름 - 간단한 설명",\n'
    "    ...\n"
    "  ],\n"
    '  "자극 신체 부위": "쉼표로 구분된 부위 목록 (ex. 어깨, 종아리, 허리)"\n'
    "}"

    "🔹 **작성 규칙**\n"
    "- 최종 출력은 반드시 JSON 객체만 포함하세요. 그 외의 주석, 설명은 절대 출력하지 마세요.\n"
    "- 목록은 너무 길지 않게 핵심 위주로 요약하세요. 단, 주요 동작은 빠짐없이 포함해야 합니다.\n"
    "- 각 동작 설명은 짧고 이해하기 쉽게 표현하고, 말투는 딱딱하지 않게 하세요.\n"
    "- 불명확한 내용은 임의로 추측하지 말고, 스크립트에 기반하여 최대한 문맥적으로 유추하세요.\n"
    "- 챗봇 인터페이스에서 **한눈에 보기 좋게** 표현하세요.\n"
)

# %%
def summarize_transcript(state: dict) -> dict:
    """
    주어진 스크립트(자막) 텍스트를 바탕으로 영상 내용을 요약합니다.
    """
    print("🚀 [Tool] summarize_transcript 호출됨")
    transcript = state["messages"][-1].content
    prompt_messages = [
        SystemMessage(content=summarize_prompt),
        HumanMessage(content=f"[분석할 스크립트]\n---\n{transcript}\n---\n\n이 영상의 내용을 분석하여 필수 JSON 형식에 맞춰 요약해주세요.")
    ]
    try:
        response = llm.invoke(prompt_messages)
        summary = response.content
        print("✅ 2. 요약 생성 성공")
        return {"messages": [AIMessage(content=summary)]}
    except Exception as e:
        error_message = f"ERROR: 요약 생성 중 오류 발생 - {e}"
        print(f"🚨 {error_message}")
        return {"messages": [AIMessage(content=error_message)]}

# %%
def route_after_transcript(state: AgentState):
    """
    [수정됨] 마지막 메시지 내용에 'ERROR:'가 있는지 확인하여 분기합니다.
    """
    last_message_content = state["messages"][-1].content
    if last_message_content.startswith("ERROR:"):
        print("🚨 오류 메시지가 감지되어 프로세스를 종료합니다.")
        return END
    else:
        print("✅ 스크립트 추출 성공. 요약 단계로 이동합니다.")
        return "summarize_transcript"

# %%
graph_builder.add_node("summarize_transcript", summarize_transcript)
graph_builder.add_node("get_youtube_transcript", get_youtube_transcript)

# %%
graph_builder.add_edge(START, "get_youtube_transcript")
graph_builder.add_conditional_edges(
    "get_youtube_transcript",
    route_after_transcript,
    {
        "summarize_transcript": "summarize_transcript",
        END: END
    }
)
graph_builder.add_edge("summarize_transcript", END)

# %%
graph = graph_builder.compile()
# graph

# %%
def run_agent(url: str):
    """
    에이전트를 실행하고 최종 결과를 스트리밍으로 출력하는 함수.
    노드 기반 그래프 구조에 맞춰, 각 단계의 메시지를 출력하고
    마지막 AIMessage가 오면 결과만 출력.
    """
    inputs = {"messages": [HumanMessage(content=url)]}
    final_state = graph.invoke(inputs)
    final_message = final_state["messages"][-1]
    content = final_message.content

    print("\n" + "="*30)
    print("✅ 최종 요약 (JSON 형식):")
    print("="*30)

    if not content or content.strip() == "":
        print("⚠️ 요약 결과가 비어 있습니다. content 값:", repr(content))
        return
    
    # LLM 응답에서 마크다운 코드 블록 제거
    if content.strip().startswith("```json"):
        start_index = content.find('{')
        end_index = content.rfind('}')
        
        if start_index != -1 and end_index != -1:
            content = content[start_index : end_index + 1]

    try:
        parsed_json = json.loads(content)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("⚠️ JSON 파싱 실패. 원본 content 출력:")
        print(content)

# # %%
# test_url = "https://youtu.be/sLe6jgHoYtk?si=BP39AJQL1PvIoWBe"
# print(f"입력 URL: {test_url}\n---")
# run_agent(test_url)

# # %%



