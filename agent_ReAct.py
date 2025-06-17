import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
import re
import requests
from typing import Any, Tuple
# import os
# print(os.getenv("ANTHROPIC_API_KEY"))

MODEL_PRICES = {
    "input": {
        "claude-3-5-sonnet-20240620": 3 / 1_000_000,
        "claude-3-7-sonnet-20250219": 3 / 1_000_000,
        "claude-sonnet-4-20250514": 3 / 1_000_000,
        "claude-opus-4-20250514" : 15 / 1_000_000
    },
    "output":{
        "claude-3-5-sonnet-20240620": 15 / 1_000_000,
        "claude-3-7-sonnet-20250219": 15 / 1_000_000,
        "claude-sonnet-4-20250514": 15 / 1_000_000,
        "claude-opus-4-20250514" : 75 / 1_000_000
    }
}

# ReAct Pattern Implementation
def parse_reason_act(response: str) -> Tuple[str, str]:
    """
    LLMの応答から推論部分と行動部分を分離する
    """
    # "Reason:"と"Act:"で分割
    reason_match = re.search(r'Reason:\s*(.*?)(?=Act:|$)', response, re.DOTALL | re.IGNORECASE)
    act_match = re.search(r'Act:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
    
    reason_part = reason_match.group(1).strip() if reason_match else ""
    act_part = act_match.group(1).strip() if act_match else response.strip()
    
    return reason_part, act_part

def search_tool(query: str) -> str:
    """
    簡単な検索ツール（実際の実装では外部APIを使用）
    """
    try:
        # DuckDuckGo Instant Answer APIを使用（簡単な例）
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('AbstractText'):
            return data['AbstractText']
        elif data.get('Answer'):
            return data['Answer']
        else:
            return f"検索結果が見つかりませんでした: {query}"
    except Exception as e:
        return f"検索エラー: {str(e)}"

def calc_tool(expression: str) -> str:
    """
    計算ツール
    """
    try:
        # 安全な計算のため、evalの代わりに制限された計算を実行
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "エラー: 許可されていない文字が含まれています"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"計算エラー: {str(e)}"

def reason_and_act(llm_call, user_input: str, tools: dict, max_iterations: int = 5) -> str:
    """
    ReActパターンの実装: Reason(思考)の出力を解析し、
    'search:' や 'calc:' といった指示があれば該当ツールを呼び出し、
    その観察結果(Observation)を次の推論に反映させる。
    """
    reasoning_log = ""
    observations_log = []
    answer = ""
    
    for iteration in range(max_iterations):
        # --- Step1: Reason（推論） ---
        # 過去の推論と観察情報を踏まえて、次に何をすべきかを考える
        prompt = f"""
You are an agent that uses the ReAct pattern (Reason-Act-Observe).
Please respond in the following format:

Reason: [Your reasoning about what to do next]
Act: [Your action - either use a tool or provide final answer]

Available tools:
- search: [query] - Search for information
- calc: [expression] - Calculate mathematical expressions
- FINISH: [final answer] - Provide the final answer

Current reasoning log:
{reasoning_log}

Current observations:
{' '.join(observations_log)}

User input: {user_input}

Please provide your reasoning and action:
        """
        
        response = llm_call(prompt)
        
        # "Reason:" と "Act:" を分割してパース
        reason_part, act_part = parse_reason_act(response)
        
        # 推論部分をログに追加
        reasoning_log += f"\nIteration {iteration + 1} - Reason: {reason_part}"
        
        # --- Step2: Act（行動） ---
        # "Act"にツール呼び出しがあるか、最終回答があるかをチェック
        if act_part.lower().startswith("search:"):
            query = act_part[len("search:"):].strip()
            # ツールを呼び出して結果を取得
            search_result = tools["search"](query)
            # --- Step3: Observation（観察） ---
            # 行動の結果を次のループの推論に反映
            observations_log.append(f"Search result for '{query}': {search_result}")
            
        elif act_part.lower().startswith("calc:"):
            expr = act_part[len("calc:"):].strip()
            calc_result = tools["calc"](expr)
            # 行動の結果をObservationとして記録
            observations_log.append(f"Calculation '{expr}': {calc_result}")
            
        elif act_part.lower().startswith("finish:"):
            # 最終回答
            answer = act_part[len("finish:"):].strip()
            break
        else:
            # "Act"が最終回答とみなし、ループを抜ける
            answer = act_part
            break
    
    # 推論ログと観察ログも含めて返す
    full_response = f"{answer}\n\n--- ReAct Process ---\n{reasoning_log}\n\nObservations:\n" + "\n".join(observations_log)
    return full_response

def init_page():
    st.set_page_config(
        page_title = "Impl AI assistant",
    )
    st.header("Impl AI assistant")
    st.sidebar.title("Options")

def init_messages():
    clear_button = st.sidebar.button("clear conversation", key ="clear")
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant")
        ]
    
    # ReActパターンの設定
    st.session_state.use_react = st.sidebar.checkbox("ReActパターンを使用", value=False)
    if st.session_state.use_react:
        st.sidebar.markdown("**ReActパターン有効**")
        # st.sidebar.markdown("- 推論→行動→観察のサイクルで動作")
        # st.sidebar.markdown("- 検索と計算ツールが利用可能")

def select_model():
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    models = ("Claude 3.5 Sonnet", "Claude 3.7 Sonnet", "Claude Sonnet 4", "Claude Opus 4")
    model = st.sidebar.radio("モデルを選択してください:", models)
    
    if model == "Claude 3.5 Sonnet":
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            temperature=temperature,
            # top_p = top_p,
            # top_k = top_k,
            model_name=st.session_state.model_name,
            max_tokens = 8192
        )
    elif model == "Claude 3.7 Sonnet":
        st.session_state.model_name = "claude-3-7-sonnet-20250219"
        return ChatAnthropic(
            temperature=temperature, 
            # top_p = top_p,
            # top_k = top_k,
            model_name=st.session_state.model_name,
            max_tokens = 32000
        )
    elif model == "Claude Sonnet 4":
        st.session_state.model_name = "claude-sonnet-4-20250514"
        return ChatAnthropic(
            temperature=temperature,
            # top_p = top_p,
            # top_k = top_k,
            model_name=st.session_state.model_name,
            max_tokens = 32000
        )
    elif model == "Claude Opus 4":
        st.session_state.model_name = "claude-opus-4-20250514"
        return ChatAnthropic(
            temperature=temperature,
            # top_p = top_p,
            # top_k = top_k,
            model_name=st.session_state.model_name,
            max_tokens = 32000
        )

def init_chain():
    llm = select_model()        
    st.session_state.llm = llm
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
        ("user", "{user_input}")
    ])
    output_parser = StrOutputParser()
    
    # Create and return the chain
    return prompt | st.session_state.llm | output_parser

def get_message_counts(text):
    try:
        return st.session_state.llm.get_num_tokens(text)
    except (AttributeError, Exception) as e:
        # Fallback to tiktoken if LLM's token counter fails
        try:
            encoding = tiktoken.encoding_for_model("cl100k_base")  # Claude models use cl100k_base encoding
            return len(encoding.encode(text))
        except Exception:
            # If all else fails, use a rough estimate (1 token ≈ 4 characters)
            return len(text) // 4

def calc_and_display_costs():
    if "llm" not in st.session_state or "model_name" not in st.session_state:
        return
        
    output_count = 0
    input_count = 0
    for role, message in st.session_state.message_history:
        token_count = get_message_counts(message)  # Count tokens
        if role == "ai":
            output_count += token_count
        else:
            input_count += token_count
    
    if len(st.session_state.message_history) == 1:
        return
    
    input_cost = MODEL_PRICES["input"][st.session_state.model_name] * input_count
    output_cost = MODEL_PRICES["output"][st.session_state.model_name] * output_count

    cost = input_cost + output_cost

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${input_cost:.5f}")
    st.sidebar.markdown(f"- Output cost: ${output_cost:.5f}")

# def self_reflective():
    

def main():
    init_page()
    init_messages()
    chain = init_chain()

    # Display chat history
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    # Monitor user input
    if user_input := st.chat_input("何を聞きたいか入力してください"):
        st.chat_message("user").markdown(user_input)
        
        # Add user message to history
        st.session_state.message_history.append(("user", user_input))
        
        # Process with the chain if it's available
        if chain is not None:
            with st.chat_message("ai"):
                if st.session_state.get("use_react", False):
                    # ReActパターンを使用
                    tools = {
                        "search": search_tool,
                        "calc": calc_tool
                    }
                    
                    def llm_call(prompt):
                        # LangChainのチェーンを使ってLLMを呼び出し
                        temp_chain = ChatPromptTemplate.from_template(prompt) | st.session_state.llm | StrOutputParser()
                        return temp_chain.invoke({})
                    
                    response = reason_and_act(llm_call, user_input, tools)
                    st.markdown(response)
                else:
                    # 通常のチャット
                    response = st.write_stream(chain.stream({"user_input": user_input}))
            
            # Add AI response to history
            st.session_state.message_history.append(("ai", response))
        else:
            with st.chat_message("ai"):
                st.error("Unable to process your request. Please check the model configuration.")

    # Calculate and display costs if possible
    try:
        calc_and_display_costs()
    except Exception as e:
        st.sidebar.error(f"Error calculating costs: {str(e)}")

if __name__ == "__main__":
    main()