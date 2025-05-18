import streamlit as st
import requests

st.set_page_config(page_title="Mental Coach", layout="wide")
st.title("How can I help you today?")

# 会话历史
if 'history' not in st.session_state:
    st.session_state.history = []

# 发送消息并更新历史
def send_message():
    user_input = st.session_state.input_text
    if not user_input:
        return
    st.session_state.history.append({"role": "user", "content": user_input})
    # 向后端请求
    res = requests.post("http://localhost:8000/chat", json={"message": user_input})
    if res.status_code == 200:
        data = res.json()
        st.session_state.history.append({"role": "assistant", "content": data["reply"]})
    else:
        st.session_state.history.append({"role": "assistant", "content": f"Error: {res.text}"})
    st.session_state.input_text = ""

# 显示对话
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Bot:** {chat['content']}")

# 输入表单
with st.form(key="input_form", clear_on_submit=True):
    st.text_input("Your message:", key="input_text")
    st.form_submit_button("Send", on_click=send_message)
