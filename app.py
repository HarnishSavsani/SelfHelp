import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.input_widget import Select, Slider
from fastapi import Request, Response

import asyncio
import logging
import os
from dotenv import load_dotenv, find_dotenv
from typing import Optional, List

from llama_index.llms.groq import Groq
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from groq import AsyncGroq

from data_layer import SQLiteDataLayer, seed_default_users
from rag_engine import RAGEngine

### Global settings
logger = logging.getLogger(__name__)
_ = load_dotenv(find_dotenv())
groq_client = AsyncGroq(max_retries=10)

# ── Context Engineering Constants ─────────────────────────────────
MAX_RECENT_MESSAGES = 10
SUMMARIZE_THRESHOLD = 15

BASE_SYSTEM_PROMPT = """\
You are Genius AI, a highly capable conversational assistant. 🧠✨

Your core capabilities:
- Answer questions thoughtfully and helpfully.
- Respond with accurate information.

Guidelines:
- Explain findings clearly with numbers, percentages, and context if applicable.
- Always be polite and professional.
- Use structured formatting (bullet points, tables where appropriate) for clarity.
- Use colorful emojis to make responses engaging and readable 🚀
- When asked about prior conversation, rely on the provided conversation summary. 📜
"""

# ── Data Layer ────────────────────────────────────────────────────
_data_layer_instance = SQLiteDataLayer()

@cl.data_layer
def get_data_layer():
    return _data_layer_instance

@cl.on_chat_start
async def start():
    """Seed default users on first launch, then handle chat start."""
    await seed_default_users(_data_layer_instance)
    await _on_chat_start()

async def _on_chat_start():
    """Handler for chat start events. Sets session variables."""

    groq_llm = Groq(model="llama-3.3-70b-versatile", temperature=0, max_retries=10)

    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has started the conversation")

    cl.user_session.set("llm", groq_llm)
    cl.user_session.set("chat_messages", [])
    cl.user_session.set("conversation_summary", "")

    # Fetch available models dynamically
    models_response = await groq_client.models.list()
    available_models = sorted([
        m.id for m in models_response.data
        if "whisper" not in m.id.lower() and "orpheus" not in m.id.lower()
        and "guard" not in m.id.lower() and "compound" not in m.id.lower()
    ])
    initial_index = (
        available_models.index("llama-3.3-70b-versatile")
        if "llama-3.3-70b-versatile" in available_models else 0
    )

    await cl.ChatSettings(
        [
            Select(
                id="LLM",
                label="Model",
                values=available_models,
                initial_index=initial_index,
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.1,
            ),
        ]
    ).send()


# ── Auth ──────────────────────────────────────────────────────────

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Password auth handler — verifies against SQLite bcrypt hashes."""
    is_valid = await _data_layer_instance.verify_password(username, password)
    if is_valid:
        role = await _data_layer_instance.get_user_role(username)
        return cl.User(identifier=username, metadata={"role": role or "USER"})
    return None


# ── Settings ──────────────────────────────────────────────────────

@cl.on_settings_update
async def setup_agent(settings):
    """Handler to manage settings updates."""
    groq_llm = Groq(
        model=settings["LLM"],
        temperature=settings["Temperature"],
        max_retries=10,
    )
    logger.info(f"New settings: LLM={settings['LLM']} | Temperature={settings['Temperature']}")
    cl.user_session.set("llm", groq_llm)


# ── Message Handler ───────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    """On message handler — handles file uploads and RAG/direct queries."""
    user = cl.user_session.get("user")
    logger.info(f"Received message: '{message.content}' from {user.identifier}")

    llm = cl.user_session.get("llm")
    thread_id = cl.context.session.thread_id

    # ── Handle file uploads ─────────────────────────────────────
    if message.elements:
        # Get or create RAG engine for this session
        rag_engine: RAGEngine = cl.user_session.get("rag_engine")
        if rag_engine is None:
            rag_engine = RAGEngine(thread_id=thread_id, llm=llm)
            cl.user_session.set("rag_engine", rag_engine)

        # Process each uploaded file
        results = []
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                file_name = element.name or os.path.basename(element.path)
                logger.info(f"Processing uploaded file: {file_name}")

                processing_msg = cl.Message(
                    content=f"⏳ Processing **{file_name}**...",
                    type="assistant_message",
                )
                await processing_msg.send()

                result = await rag_engine.ingest_file(element.path, file_name)
                results.append(result)

                # Update the processing message with the result
                processing_msg.content = result
                await processing_msg.update()

        # If user also included a text question with files, answer it
        if message.content.strip():
            await generate_rag_answer(message.content)
        return

    # ── Handle queries (RAG or direct) ──────────────────────────
    rag_engine: RAGEngine = cl.user_session.get("rag_engine")
    if rag_engine and rag_engine.has_data():
        await generate_rag_answer(message.content)
    else:
        await generate_answer(message.content)


# ── Lifecycle Handlers ────────────────────────────────────────────

@cl.on_stop
async def on_stop():
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} stopped the task.")
    await cl.Message("You have stopped the task!").send()


@cl.on_chat_end
def on_chat_end():
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} ended the chat")
    # Clean up RAG engine connections
    rag_engine: RAGEngine = cl.user_session.get("rag_engine")
    if rag_engine:
        rag_engine.close()
        cl.user_session.set("rag_engine", None)


@cl.on_logout
def on_logout(request: Request, response: Response):
    logger.info("Clearing cookies...")
    for cookie_name in request.cookies.keys():
        response.delete_cookie(cookie_name)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume a chat — restore messages and RAG engine."""
    groq_llm = Groq(model="llama-3.3-70b-versatile", temperature=0, max_retries=10)

    # ── Restore conversation history ────────────────────────────
    chat_messages = []
    for message in thread.get("steps", []):
        if message["type"] == "user_message":
            chat_messages.append({"role": "user", "content": message["output"]})
        elif message["type"] == "assistant_message":
            chat_messages.append({"role": "assistant", "content": message["output"]})

    cl.user_session.set("chat_messages", chat_messages)
    cl.user_session.set("conversation_summary", "")
    cl.user_session.set("llm", groq_llm)

    # Summarize old messages if needed
    if len(chat_messages) > MAX_RECENT_MESSAGES:
        summary = await _summarize_messages(groq_llm, chat_messages[:-MAX_RECENT_MESSAGES])
        cl.user_session.set("conversation_summary", summary)

    # ── Restore RAG engine if data exists ───────────────────────
    thread_id = thread.get("id", "")
    rag_engine = RAGEngine.load_from_storage(thread_id, groq_llm)
    if rag_engine:
        cl.user_session.set("rag_engine", rag_engine)
        files_summary = rag_engine.get_loaded_files_summary()
        logger.info(f"RAG engine restored for thread {thread_id}")
    else:
        cl.user_session.set("rag_engine", None)
        files_summary = None

    user = cl.user_session.get("user")
    logger.info(f"{user} resumed chat")

    welcome_content = (
        "👋 **Welcome back!** Your conversation history has been restored.\n"
    )
    if files_summary:
        welcome_content += (
            f"\n📎 **Your uploaded data is ready to query:**\n{files_summary}\n"
        )
    welcome_content += "\nFeel free to continue chatting!"

    await cl.Message(content=welcome_content).send()


# ── Context Engineering ───────────────────────────────────────────

async def _summarize_messages(llm, messages: List[dict]) -> str:
    """Use the LLM to create a concise summary of older conversation messages."""
    conversation_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )
    summary_prompt = (
        "Summarize the following conversation concisely, preserving key facts, "
        "decisions, user preferences, and important context. "
        "Be thorough but brief:\n\n"
        f"{conversation_text}\n\nSummary:"
    )
    response = llm.complete(summary_prompt)
    return str(response).strip()


def _build_context_messages(
    summary: str,
    recent_messages: List[dict],
) -> List[ChatMessage]:
    """Build the final message list for the LLM with context engineering."""
    messages = []
    system_content = BASE_SYSTEM_PROMPT

    if summary:
        system_content += f"\n\n## Previous Conversation Summary\n{summary}"

    messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content))

    for m in recent_messages:
        role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=m["content"]))

    return messages


# ── Answer Generation ─────────────────────────────────────────────

async def generate_answer(query: str):
    """Generate a streamed response from the direct LLM."""
    chat_messages: List[dict] = cl.user_session.get("chat_messages", [])
    summary: str = cl.user_session.get("conversation_summary", "")
    llm = cl.user_session.get("llm")

    msg = cl.Message("", type="assistant_message")

    chat_messages.append({"role": "user", "content": query})

    # Summarize older messages if conversation is long
    if len(chat_messages) > SUMMARIZE_THRESHOLD:
        older_messages = chat_messages[:-MAX_RECENT_MESSAGES]
        if summary:
            older_messages = [{"role": "assistant", "content": f"Previous summary: {summary}"}] + older_messages
        summary = await _summarize_messages(llm, older_messages)
        chat_messages = chat_messages[-MAX_RECENT_MESSAGES:]
        cl.user_session.set("conversation_summary", summary)

    # Build context-aware message list
    context_messages = _build_context_messages(summary, chat_messages)

    # Direct LLM chat
    response = await llm.astream_chat(context_messages)
    response_str = ""
    async for token in response:
        if token.delta:
            await msg.stream_token(token.delta)
            response_str += token.delta

    await msg.send()

    chat_messages.append({"role": "assistant", "content": response_str})
    cl.user_session.set("chat_messages", chat_messages)

    return msg


async def generate_rag_answer(query: str):
    """Generate a response using the RAG engine (structured/unstructured)."""
    rag_engine: RAGEngine = cl.user_session.get("rag_engine")
    chat_messages: List[dict] = cl.user_session.get("chat_messages", [])

    msg = cl.Message("", type="assistant_message")

    chat_messages.append({"role": "user", "content": query})

    try:
        # Query through the RAG router
        answer = await rag_engine.query(query, chat_messages)
        msg.content = answer
        await msg.send()
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        msg.content = (
            "❌ An error occurred while processing your query. "
            "Please try rephrasing your question."
        )
        await msg.send()
        answer = msg.content

    chat_messages.append({"role": "assistant", "content": answer})
    cl.user_session.set("chat_messages", chat_messages)

    return msg
