from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.tools import tool, Tool
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description_and_args
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from vector_store import VectorStore
from web_search import WebSearch
import config
import json
from typing import List, Dict, Any

class RAGAgent:
    def __init__(self, session_id: str = "default"):
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
        self.vector_store = VectorStore()
        self.web_search = WebSearch()
        self.session_id = session_id
        
        # Initialize memory with window to keep last 10 exchanges (20 messages)
        self.memory = ConversationBufferWindowMemory(
            k=20,  # Keep last 20 messages (10 exchanges)
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        self.agent_executor = None
        self._setup_tools()
        self._create_agent()
    
    def _setup_tools(self):
        """Setup tools for vector search and web search"""
        
        @tool
        def vector_search_tool(query: str) -> str:
            """Tool for searching the vector store."""
            if not self.vector_store.is_available():
                return "Vector store is not available. Please upload documents first."
            
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm, 
                    retriever=self.vector_store.retriever
                )
                return qa_chain.run(query)
            except Exception as e:
                return f"Vector search error: {str(e)}"
        
        @tool
        def web_search_tool_func(query: str) -> str:
            """Tool for performing web search."""
            return self.web_search.search(query)
        
        self.tools = [
            Tool(
                name="VectorStoreSearch",
                func=vector_search_tool,
                description="Use this to search the vector store for information about uploaded documents."
            ),
            Tool(
                name="WebSearch",
                func=web_search_tool_func,
                description="Use this to perform a web search for current information."
            ),
        ]
    
    def _create_agent(self):
        """Create the RAG agent with memory"""
        system_prompt = """You are a helpful AI assistant that can search through uploaded documents and the web to answer questions. You have access to the following tools: {tools}

IMPORTANT INSTRUCTIONS:
1. Always consider the conversation history when answering questions
2. If the user refers to something from earlier in the conversation, use that context
3. Always try the "VectorStoreSearch" tool first for document-related questions
4. Only use "WebSearch" if the vector store does not contain the required information
5. Be conversational and remember what was discussed previously

Previous conversation:
{chat_history}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:
Question: input question to answer
Thought: consider previous conversation and current question, then decide on next steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action."""

        human_prompt = """{input}
{agent_scratchpad}
(reminder to always respond in a JSON blob)"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
        
        prompt = prompt.partial(
            tools=render_text_description_and_args(list(self.tools)),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        
        chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
                chat_history=lambda x: self._format_chat_history()
            )
            | prompt
            | self.llm
            | JSONAgentOutputParser()
        )
        
        self.agent_executor = AgentExecutor(
            agent=chain,
            tools=self.tools,
            memory=self.memory,
            handle_parsing_errors=True,
            verbose=False
        )
    
    def _format_chat_history(self) -> str:
        """Format chat history for the prompt"""
        if not self.memory.chat_memory.messages:
            return "No previous conversation."
        
        formatted_history = []
        for message in self.memory.chat_memory.messages[-10:]:  # Last 5 exchanges
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted_history) if formatted_history else "No previous conversation."
    
    def query(self, question: str) -> str:
        """Process a query through the RAG agent with memory"""
        try:
            result = self.agent_executor.invoke({"input": question})
            return result['output']
        except Exception as e:
            return f"Agent error: {str(e)}"
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
    
    def get_memory_summary(self) -> List[Dict[str, str]]:
        """Get a summary of the conversation history"""
        messages = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"role": "assistant", "content": message.content})
        return messages
    
    def load_memory_from_messages(self, messages: List[Dict[str, str]]):
        """Load memory from a list of message dictionaries"""
        self.memory.clear()
        for message in messages:
            if message["role"] == "user":
                self.memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                self.memory.chat_memory.add_ai_message(message["content"])
    
    def load_documents(self, file_path: str) -> bool:
        """Load documents into vector store"""
        try:
            documents = self.vector_store.load_documents(file_path)
            self.vector_store.create_vectorstore(documents)
            self.vector_store.save_vectorstore()
            return True
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False
    
    def load_existing_vectorstore(self) -> bool:
        """Load existing vector store"""
        return self.vector_store.load_vectorstore()