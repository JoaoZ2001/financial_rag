from collections import deque
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from src.config import LM_STUDIO_URL, LM_STUDIO_API_KEY, MODEL_NAME
from src.vector_store import retriever_amd, retriever_intel, retriever_nvidia

# --- Contextualize Query Prompt ---
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# --- Memory Chain Class ---
class MemoryChain:
    """RAG Chain with sliding window conversational memory and contextual retrieval."""
    
    def __init__(self, retriever, prompt, llm, k=5):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm
        self.parser = StrOutputParser()
        self.memory = deque(maxlen=k)  # Stores last k interactions
        
        # Chain for reformulating questions
        self.history_aware_retriever = (
            contextualize_q_prompt | llm | StrOutputParser()
        )
    
    def _get_chat_history(self):
        """Converts internal memory to LangChain message format."""
        messages = []
        for human_msg, ai_msg in self.memory:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages
    
    def _contextualize_question(self, question: str, chat_history: list) -> str:
        """Reformulates question if history exists."""
        if not chat_history:
            return question
            
        # Invoke the history aware retriever to rephrase the question
        return self.history_aware_retriever.invoke({
            "chat_history": chat_history,
            "question": question
        })
    
    def invoke(self, question: str) -> str:
        """Process question with contextual retrieval and existing RAG logic."""
        chat_history = self._get_chat_history()
        
        # 1. Reformulate question based on history
        contextualized_question = self._contextualize_question(question, chat_history)
        
        # 2. Retrieve documents using focused question
        docs = self.retriever.invoke(contextualized_question)
        
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        context = format_docs(docs)
        
        # 3. Generate answer using ORIGINAL question + context + history
        # (Using original question preserves user's tone, while context is now relevant)
        prompt_value = self.prompt.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question 
        })
        
        response = self.llm.invoke(prompt_value)
        answer = self.parser.invoke(response)
        
        self.memory.append((question, answer))
        return answer

    def stream(self, question: str):
        """Stream the response with contextual retrieval."""
        chat_history = self._get_chat_history()
        
        # 1. Reformulate
        contextualized_question = self._contextualize_question(question, chat_history)
        
        # 2. Retrieve
        docs = self.retriever.invoke(contextualized_question)
        
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
            
        context = format_docs(docs)
        
        # 3. Generate
        prompt_value = self.prompt.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question
        })
        
        full_response = ""
        for chunk in self.llm.stream(prompt_value):
            content = self.parser.invoke(chunk)
            full_response += content
            yield content
            
        self.memory.append((question, full_response))

    def clear_memory(self):
        self.memory.clear()

def build_10k_prompt(company: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a {company} 10-K analyst. Use ONLY the context below. Be concise.

    Rules:
    - Numbers: include units ($M, $B) and fiscal year. Use tables when comparing multiple values.
    - Text: quote directly from context. Use bullet points only for lists of 3+ items.
    - Missing info: "Not available in 10-K."
    - No explanations unless asked.

    Context: {{context}}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

amd_prompt    = build_10k_prompt("AMD")
intel_prompt  = build_10k_prompt("Intel")
nvidia_prompt = build_10k_prompt("NVIDIA")

# --- Factory Function ---
def get_llm():
    return ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=0.1,
        model=MODEL_NAME,
        max_tokens=512
    )

def create_agent(company: str):
    """Creates a fresh agent instance for the specified company."""
    llm = get_llm()
    # Note: k=5 is default in __init__ but being explicit here is fine
    if company.lower() == 'amd':
        return MemoryChain(retriever_amd, amd_prompt, llm, k=5)
    elif company.lower() == 'intel':
        return MemoryChain(retriever_intel, intel_prompt, llm, k=5)
    elif company.lower() == 'nvidia':
        return MemoryChain(retriever_nvidia, nvidia_prompt, llm, k=5)
    else:
        raise ValueError(f"Unknown company: {company}")
