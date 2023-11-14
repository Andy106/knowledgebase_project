from core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Anand's Medium Blog - Helper Bot")


prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if "user_prompt_history" not in st.session_state:  # Initialize streamlit session state with empty lists. If user_prompt_history key 
# does not exist in session state, initialize it by setting it to an empty list.
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


if prompt:
    with st.spinner("Generating response.."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"]) # The actual trigger of the ConversationalRetrievalchain

        # generated_response = run_llm(query=prompt) # The actual trigger of the ConversationalRetrievalchain

    #    print(generated_response)

        formatted_response = (
            f"{generated_response}"
        )

        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['answer']} \n\n {sources}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response['answer'])) # Store prompt and response as a tuple under 
        # chat_history key in the session state (to pass on to the LLM)

if st.session_state["chat_answers_history"]:   
    for generated_response, user_query in zip(   # Iterate over the chat history using zip function
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)