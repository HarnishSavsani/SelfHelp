import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.input_widget import Select, Switch, Slider
from fastapi import Request, Response

import logging
import os
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict
from typing import Optional



from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, ToolCall
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from groq import AsyncGroq

### Global settings
logger = logging.getLogger(__name__)
_ = load_dotenv(find_dotenv())
groq_client = AsyncGroq()
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
SYSTEM_PROMPTS = {
    "The Assistant": "You are a helpful AI assistant.",
}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Password auth handler for login"""
    
    users = {
        "admin": ("admin", "ADMIN"),
        "Harnish": ("Pass@1234", "USER"),
        "Hrishikesh": ("Pass@1234", "USER"),
        "Sarvesh": ("Pass@1234", "USER"),
        "Aniket": ("Pass@1234", "USER"),
        "Avnish": ("Pass@1234", "USER")
    }
    
    if username in users and users[username][0] == password:
        return cl.User(identifier=username, metadata={"role": users[username][1]})
        
    return None

@cl.set_chat_profiles
async def chat_profile():
    """Chat profile setter."""
    
    return [
        cl.ChatProfile(
            name="The Assistant",
            markdown_description="This Genius AI is your personal assistant",
            icon = "public/icon.png"
        )
    ]

@cl.on_chat_start
async def start():
    """Handler for chat start events. Sets session variables."""
    
    groq_llm = Groq(model="llama-3.3-70b-versatile", temperature=0)
    agent = FunctionAgent(tools=[],llm=groq_llm,)
    chat_profile = cl.user_session.get("chat_profile")
    if not chat_profile:
        chat_profile = "The Assistant"
        
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has started the conversation")
    
    cl.user_session.set("llm", groq_llm)
    cl.user_session.set("agent_tools", [])
    cl.user_session.set("context", Context(agent))
    cl.user_session.set("agent", agent)
    
    system_prompt = SYSTEM_PROMPTS[chat_profile]
    memory = ChatMemoryBuffer.from_defaults()
    memory.put(
        ChatMessage(
            role=MessageRole.SYSTEM, 
            content=system_prompt
        )
    )
    cl.user_session.set("memory", memory)
    
    models_response = await groq_client.models.list()
    available_models = [m.id for m in models_response.data if "whisper" not in m.id.lower()]
    initial_index = available_models.index("llama-3.3-70b-versatile") if "llama-3.3-70b-versatile" in available_models else 0
    
    settings = await cl.ChatSettings(
        [            
            Select(
                id="LLM",
                label="Groq model to use",
                values=available_models,
                initial_index=initial_index,
            ),
            Switch(
                id="Greet_on_message",
                label="Greet user when message is received",
                initial=False,
            ),
            Slider(
                id="Temperature",
                label="Temperature of the LLM",
                initial=0,
                min=0,
                max=1,
                step=0.1
            )
        ]
    ).send()

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
    
    cl.user_session.set("greet", settings["Greet_on_message"])
    

@cl.on_message
async def on_message(message: cl.Message):
    """On message handler to handle message received events"""
    
    user = cl.user_session.get("user")
    logger.info(f"Received message: '{message.content}' from {user.identifier}")
    
    greet = cl.user_session.get("greet")
    if greet is True:
        await cl.Message(f"Hello there {user.identifier}!").send()
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
            name = "_".join(str(name).split(" ")),
            description=str(description)
        )
        agent_tools = cl.user_session.get("agent_tools", [])
        agent_tools.append(tool)
        
        agent = FunctionAgent(tools=agent_tools, llm=groq_llm)
        cl.user_session.set("agent", agent)
        cl.user_session.set("agent_tools", agent_tools)
    
    reply = await generate_answer(message.content)
    
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
    ### Handler to tidy up resources
    logger.info("Clearing cookies...")
    for cookie_name in request.cookies.keys():
        response.delete_cookie(cookie_name)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Handler function to resume a chat"""
    
    ## Setup LLM
    groq_llm = Groq(model="llama-3.3-70b-versatile", temperature=0)
    
    ## Restore memory buffer
    memory = ChatMemoryBuffer.from_defaults()
    root_messages = [m for m in thread["steps"]]
    for message in root_messages:
        print(message)
        if message["type"] == "user_message":
            memory.put(
                ChatMessage(
                    role=MessageRole.USER,
                    content=message['output']
                )
            )
        else:
            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=message['output']
                )
            )
    cl.user_session.set("memory", memory)
    
    # ## Restore agent
    agent_tools = []
    
    agent = FunctionAgent(
        tools=agent_tools,
        llm=groq_llm,
    )
    cl.user_session.set("agent", agent)
    cl.user_session.set("context", Context(agent))
    
    user = cl.user_session.get("user")
    logger.info(f"{user} has resumed chat")
    await cl.Message("Chat resumed. Do note that previously uploaded documents will not be available in this chat and must be uploaded again").send()
    

## Steps
## Utility functions
async def generate_answer(query: str):
    agent = cl.user_session.get("agent")
    memory = cl.user_session.get("memory")
    chat_history = memory.get()
    agent_tools = cl.user_session.get("agent_tools", [])
    msg = cl.Message("", type="assistant_message")
    
    if len(agent_tools) > 0:
        context = cl.user_session.get("context")
        handler = agent.run(
            query, 
            chat_history = chat_history,
            ctx = context
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
        llm = cl.user_session.get("llm")
        temp_history = chat_history + [ChatMessage(role=MessageRole.USER, content=query)]
        response = await llm.astream_chat(temp_history)
        
        response_str = ""
        async for token in response:
            if token.delta:
                await msg.stream_token(token.delta)
                response_str += token.delta
                
    await msg.send()
    memory.put(
        ChatMessage(
            role = MessageRole.USER,
            content= query
        )
    )
    memory.put(
        ChatMessage(
            role = MessageRole.ASSISTANT,
            content = response_str
        )
    )
    cl.user_session.set("memory", memory)
    return msg

