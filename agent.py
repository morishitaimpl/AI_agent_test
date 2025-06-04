import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
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
            model_name=st.session_state.model_name
        )
    elif model == "Claude 3.7 Sonnet":
        st.session_state.model_name = "claude-3-7-sonnet-20250219"
        return ChatAnthropic(
            temperature=temperature, 
            # top_p = top_p,
            # top_k = top_k,
            model_name=st.session_state.model_name
        )
    elif model == "Claude Sonnet 4":
        st.session_state.model_name = "claude-sonnet-4-20250514"
        return ChatAnthropic(
            temperature=temperature,
            # top_p = top_p,
            # top_k = top_k,
            model_name=st.session_state.model_name
        )
    elif model == "Claude Opus 4":
        st.session_state.model_name = "claude-opus-4-20250514"
        return ChatAnthropic(
            temperature=temperature,
            # top_p = top_p,
            # top_k = top_k,
            model_name=st.session_state.model_name
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