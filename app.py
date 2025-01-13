import os
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from rag import upload_and_process_pdf, rag_answer

# 设置Streamlit页面标题和布局
st.set_page_config(page_title="PDF文档问答系统", page_icon="📄", layout="wide")
st.title('📄 PDF文档问答系统')

# 自定义CSS样式
st.markdown(
    """
    <style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    .stButton button {
    background-color: #1E88E5;
    color: white;
    border-radius: 5px;
    padding: 5px 10px; /* Reduced padding */
    font-size: 16px;
    cursor: pointer;
    }
    .stButton button:hover {
        background-color: #1976D2;
    }
    .stChatInput textarea {
        border-radius: 20px;
        padding: 10px;
        border: 1px solid #E0E0E0;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 10px 0;
        max-width: 60%;
    }
    .stChatMessage.user {
        background-color: #E0E0E0;
        align-self: flex-end;
    }
    .stChatMessage.assistant {
        background-color: #DCF8C3;
        align-self: flex-start;
    }
    .stSidebar {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 初始化会话状态
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 侧边栏
with st.sidebar:
    # 上传和处理PDF文档
    with st.expander("上传和处理PDF文档"):
        st.markdown("### 📤 上传和处理PDF文档")
        uploaded_file = st.file_uploader("选择PDF文件", type="pdf", key="uploaded_file")
        if uploaded_file and st.button("处理PDF文档", key="process_pdf"):
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("正在处理PDF文档..."):
                st.session_state.vectorstore = upload_and_process_pdf("uploaded_file.pdf")
            st.success("PDF文档已上传并处理完成！")
        if st.button("重新上传PDF文档", key="reupload_pdf"):
            st.session_state.vectorstore = None
            st.warning("请重新上传并处理新的PDF文档。")

    # 旧的对话
    with st.expander("旧的对话"):
        if st.session_state.chat_sessions:
            for session_id, session_data in st.session_state.chat_sessions.items():
                col1, col2 = st.columns([11, 2])  # Adjusted column ratio
                if col1.button(f"会话 {session_id}", key=f"switch_{session_id}"):
                    st.session_state.current_session_id = session_id
                if col2.button("🗑️", key=f"delete_{session_id}"):
                    del st.session_state.chat_sessions[session_id]
                    if st.session_state.current_session_id == session_id:
                        st.session_state.current_session_id = None
                    st.rerun()
        else:
            st.write("暂无旧的对话。")

    # 对话管理
    with st.expander("对话管理"):
        st.markdown("---")
        if st.button("创建新对话", key="new_chat"):
            new_session_id = len(st.session_state.chat_sessions) + 1
            st.session_state.chat_sessions[new_session_id] = {"messages": []}
            st.session_state.current_session_id = new_session_id
            st.rerun()
        if st.session_state.current_session_id and st.button("清空当前对话", key="clear_chat"):
            st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = []
            st.rerun()

# 主内容区域
with st.container():
    col1, col2 = st.columns([3, 2])
    with col1:
        if st.session_state.current_session_id:
            current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
            for message in current_session["messages"]:
                with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])
            if prompt := st.chat_input("请输入您的问题："):
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(prompt)
                current_session["messages"].append({"role": "user", "content": prompt})
                with st.spinner("正在生成回答..."):
                    if st.session_state.vectorstore:
                        answer = rag_answer(prompt, st.session_state.vectorstore)
                    else:
                        model = ChatOllama(model="llama3.2:3b")
                        response = model.invoke([HumanMessage(content=prompt)])
                        answer = response.content
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(answer)
                current_session["messages"].append({"role": "assistant", "content": answer})
        else:
            st.info("请创建一个新的对话会话以开始提问。")

    with col2:
        # 提示信息
        st.markdown("### 📝 提示信息")
        st.markdown(
            """
            <div class="card">
                <p>欢迎使用PDF文档问答系统！您可以通过以下步骤开始：</p>
                <ol>
                    <li>上传PDF文档并点击“处理PDF文档”按钮。</li>
                    <li>创建新的对话会话。</li>
                    <li>输入您的问题，系统会从文档中检索答案或使用模型推理回答。</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 示例问题
        st.markdown("### 💡 示例问题")
        st.markdown(
            """
            <div class="card">
                <p>以下是一些您可以尝试的示例问题：</p>
                <ul>
                    <li>文档中提到了哪些关键数据？</li>
                    <li>请总结文档的主要内容。</li>
                    <li>文档中是否有关于[具体主题]的信息？</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 系统功能
        st.markdown("### 📚 系统功能")
        st.markdown(
            """
            <div class="card">
                <ul>
                    <li>**文档问答**：从上传的PDF文档中检索答案。</li>
                    <li>**模型推理**：如果文档中没有相关内容，系统会使用模型进行推理回答。</li>
                    <li>**多会话管理**：支持创建多个对话会话，并随时切换。</li>
                    <li>**清空对话**：可以清空当前会话的聊天记录。</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )