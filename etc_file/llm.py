import re
import json
from etc_file.comment_agent import graph as comment_agent
from script_agent_04 import graph as script_agent
from langchain_core.messages import HumanMessage

def extract_youtube_url(text):
    """텍스트에서 유튜브 URL을 추출합니다."""
    match = re.search(r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)', text)
    return match.group(0) if match else None

def _clean_and_parse_json(content: str) -> dict:
    """LLM의 응답에서 마크다운을 제거하고 JSON으로 파싱합니다."""
    if content.strip().startswith("```json"):
        start_index = content.find('{')
        end_index = content.rfind('}')
        if start_index != -1 and end_index != -1:
            content = content[start_index : end_index + 1]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 파싱 실패 시 원본 텍스트나 에러 메시지를 포함한 객체를 반환할 수 있습니다.
        return {"error": "JSON 파싱에 실패했습니다.", "original_content": content}

def get_ai_message_v2(user_message: str):
    """
    [스트리밍 버전] 사용자 메시지를 분석하여,
    완료되는 작업의 결과를 순차적으로 yield 합니다.
    """
    youtube_url = extract_youtube_url(user_message)
    if not youtube_url:
        yield {"error": "메시지에서 유튜브 URL을 찾을 수 없습니다."}
        return

    # 1. 스크립트 요약 처리 및 결과 반환
    if "스크립트" in user_message:
        print("🤖 스크립트 요약을 시작합니다...")
        result = {}
        try:
            script_input = {"messages": [HumanMessage(content=youtube_url)]}
            final_script_state = script_agent.invoke(script_input)
            script_content = final_script_state['messages'][-1].content
            result['script_summary'] = _clean_and_parse_json(script_content)
        except Exception as e:
            result['script_summary'] = {"error": f"스크립트 에이전트 실행 중 오류 발생: {e}"}
        yield result

    # 2. 댓글 요약 처리 및 결과 반환
    if "댓글" in user_message:
        print("🤖 댓글 요약을 시작합니다...")
        result = {}
        try:
            comment_input = {"messages": [HumanMessage(content=youtube_url)]}
            final_comment_state = comment_agent.invoke(comment_input)
            comment_content = final_comment_state['messages'][-1].content
            result['comment_summary'] = comment_content
        except Exception as e:
            result['comment_summary'] = f"댓글 에이전트 실행 중 오류 발생: {e}"
        yield result