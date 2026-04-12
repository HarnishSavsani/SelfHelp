import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.input_widget import Select, Switch, Slider
from fastapi import Request, Response

import logging
import os
from dotenv import load_dotenv, find_dotenv
from typing import Optional, List

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, ToolCall
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from groq import AsyncGroq

from data_layer import SQLiteDataLayer, seed_default_users

### Global settings
logger = logging.getLogger(__name__)
_ = load_dotenv(find_dotenv())
groq_client = AsyncGroq()
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ── Context Engineering Constants ─────────────────────────────────
MAX_RECENT_MESSAGES = 10        # Keep the last N messages in full
SUMMARIZE_THRESHOLD = 15        # Start summarizing when total messages exceed this
SYSTEM_PROMPT = """You are Genius AI, a highly capable personal assistant. \
You provide clear, accurate, and thoughtful answers. \
When asked about prior conversation, rely on the provided conversation summary for context."""

# ── Data Layer ────────────────────────────────────────────────────
_data_layer_instance = SQLiteDataLayer()

@cl.data_layer
def get_data_layer():
    return _data_layer_instance


@cl.on_chat_start
async def start():
    """Seed default users on first launch, then handler for chat start."""
    await seed_default_users(_data_layer_instance)
    await _on_chat_start()


async def _on_chat_start():
    """Handler for chat start events. Sets session variables."""

    groq_llm = Groq(model="llama-3.3-70b-versatile", temperature=0)
    agent = FunctionAgent(tools=[], llm=groq_llm)

    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has started the conversation")

    cl.user_session.set("llm", groq_llm)
    cl.user_session.set("agent_tools", [])
    cl.user_session.set("context", Context(agent))
    cl.user_session.set("agent", agent)

    # Context engineering: store messages as a list of dicts
    cl.user_session.set("chat_messages", [])
    cl.user_session.set("conversation_summary", "")

    # Fetch available models dynamically
    models_response = await groq_client.models.list()
    available_models = sorted([
        m.id for m in models_response.data
        if "whisper" not in m.id.lower() and "orpheus" not in m.id.lower()
        and "guard" not in m.id.lower() and "compound" not in m.id.lower()
    ])
    initial_index = available_models.index("llama-3.3-70b-versatile") if "llama-3.3-70b-versatile" in available_models else 0

    settings = await cl.ChatSettings(
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
                step=0.1
            )
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
    """Handler to manage settings updates"""

    groq_llm = Groq(
        model=settings["LLM"],
        temperature=settings["Temperature"]
    )
    logger.info(f"New settings received. LLM: {settings['LLM']} | Temperature: {settings['Temperature']}")
    cl.user_session.set("llm", groq_llm)

    agent_tools = cl.user_session.get("agent_tools")
    agent = FunctionAgent(tools=agent_tools, llm=groq_llm)
    logger.info("Agent instantiated")
    cl.user_session.set("agent", agent)


# ── Message Handler ───────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    """On message handler to handle message received events"""

    user = cl.user_session.get("user")
    logger.info(f"Received message: '{message.content}' from {user.identifier}")

    if len(message.elements) > 0:
        ## Builds an in-memory RAG engine
        await cl.Message("Processing files").send()
        filepaths = [file.path for file in message.elements]
        filenames = [file.name for file in message.elements]
        logger.info(f"filepaths: {filepaths}")
        logger.info(f"filenames: {filenames}")

        ## Convert uploaded documents to LlamaIndex Document objects
        documents = SimpleDirectoryReader(input_files=filepaths).load_data()

        ## Ingest documents into in-memory Vector Database.
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        await cl.Message("Processed uploaded files").send()

        groq_llm = cl.user_session.get("llm")
        name = groq_llm.complete(f"Based on these filenames, come up with a short, concise name that describes these documents. For example 'MBA Value Analysis'. Do not return any '.pdf' or file extensions, just the name. Filenames: {', '.join(filenames)}")
        description = groq_llm.complete(f"Based on these filenames, come up with a consolidated description that describes these documents. For example 'Answers questions about animals'. Filenames: {', '.join(filenames)}")
        await cl.Message(f"Uploaded document/s follow the theme: {name}. Here's the general description of the document/s uploaded: {description}").send()

        tool = QueryEngineTool.from_defaults(
            query_engine=index.as_query_engine(similarity_top_k=8, llm=groq_llm),
            name="_".join(str(name).split(" ")),
            description=str(description)
        )
        agent_tools = cl.user_session.get("agent_tools", [])
        agent_tools.append(tool)

        agent = FunctionAgent(tools=agent_tools, llm=groq_llm)
        cl.user_session.set("agent", agent)
        cl.user_session.set("agent_tools", agent_tools)

    reply = await generate_answer(message.content)


# ── Lifecycle Handlers ────────────────────────────────────────────

@cl.on_stop
async def on_stop():
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has stopped the task!")
    await cl.Message("You have stopped the task!").send()

@cl.on_chat_end
def on_chat_end():
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has ended the chat")

@cl.on_logout
def on_logout(request: Request, response: Response):
    logger.info("Clearing cookies...")
    for cookie_name in request.cookies.keys():
        response.delete_cookie(cookie_name)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Handler function to resume a chat — restores context from persisted steps."""

    groq_llm = Groq(model="llama-3.3-70b-versatile", temperature=0)

    # Restore chat messages from thread steps
    chat_messages = []
    root_messages = [m for m in thread["steps"]]
    for message in root_messages:
        if message["type"] == "user_message":
            chat_messages.append({"role": "user", "content": message["output"]})
        elif message["type"] == "assistant_message":
            chat_messages.append({"role": "assistant", "content": message["output"]})

    cl.user_session.set("chat_messages", chat_messages)
    cl.user_session.set("conversation_summary", "")
    cl.user_session.set("llm", groq_llm)

    # If there are many messages, summarize the older ones
    if len(chat_messages) > MAX_RECENT_MESSAGES:
        summary = await _summarize_messages(groq_llm, chat_messages[:-MAX_RECENT_MESSAGES])
        cl.user_session.set("conversation_summary", summary)

    agent_tools = []
    agent = FunctionAgent(tools=agent_tools, llm=groq_llm)
    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_tools", agent_tools)
    cl.user_session.set("context", Context(agent))

    user = cl.user_session.get("user")
    logger.info(f"{user} has resumed chat")
    await cl.Message("Chat resumed. Previously uploaded documents must be uploaded again.").send()


# ── Context Engineering ───────────────────────────────────────────

async def _summarize_messages(llm, messages: List[dict]) -> str:
    """Use the LLM to create a concise summary of older conversation messages."""
    conversation_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )

    summary_prompt = f"""Summarize the following conversation concisely, preserving key facts, \
decisions, user preferences, and important context. Be thorough but brief:

{conversation_text}

Summary:"""

    response = llm.complete(summary_prompt)
    return str(response).strip()


def _build_context_messages(
    summary: str,
    recent_messages: List[dict],
) -> List[ChatMessage]:
    """Build the final message list for the LLM using context engineering.

    Structure:
        1. System prompt (with embedded summary of older conversation)
        2. Recent messages (last N turns in full)
    """
    messages = []

    # System prompt with optional conversation summary
    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\n## Previous Conversation Summary\n{summary}"

    messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content))

    # Recent messages in full
    for m in recent_messages:
        role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=m["content"]))

    return messages


# ── Answer Generation ─────────────────────────────────────────────

async def generate_answer(query: str):
    agent = cl.user_session.get("agent")
    agent_tools = cl.user_session.get("agent_tools", [])
    chat_messages: List[dict] = cl.user_session.get("chat_messages", [])
    summary = cl.user_session.get("conversation_summary", "")
    llm = cl.user_session.get("llm")

    msg = cl.Message("", type="assistant_message")

    # Add the user's query to our message history
    chat_messages.append({"role": "user", "content": query})

    # Check if we need to summarize older messages
    if len(chat_messages) > SUMMARIZE_THRESHOLD:
        older_messages = chat_messages[:-MAX_RECENT_MESSAGES]
        # Combine existing summary with newly overflowed messages
        if summary:
            older_messages = [{"role": "assistant", "content": f"Previous summary: {summary}"}] + older_messages
        summary = await _summarize_messages(llm, older_messages)
        # Keep only recent messages
        chat_messages = chat_messages[-MAX_RECENT_MESSAGES:]
        cl.user_session.set("conversation_summary", summary)

    # Build context-engineered message list
    context_messages = _build_context_messages(summary, chat_messages)

    if len(agent_tools) > 0:
        context = cl.user_session.get("context")
        handler = agent.run(
            query,
            chat_history=context_messages,
            ctx=context
        )
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                await msg.stream_token(event.delta)
            elif isinstance(event, ToolCall):
                with cl.Step(name=f"{event.tool_name} tool", type="tool"):
                    continue

        response = await handler
        response_str = str(response)
    else:
        # Direct LLM streaming (no tools)
        temp_history = context_messages + [ChatMessage(role=MessageRole.USER, content=query)]
        response = await llm.astream_chat(temp_history)

        response_str = ""
        async for token in response:
            if token.delta:
                await msg.stream_token(token.delta)
                response_str += token.delta

    await msg.send()

    # Store assistant response in our message history
    chat_messages.append({"role": "assistant", "content": response_str})
    cl.user_session.set("chat_messages", chat_messages)

    return msg
