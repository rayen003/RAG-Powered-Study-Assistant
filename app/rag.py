from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from config import OPENAI_API_KEY, PERSIST_DIRECTORY
import numpy as np
from .tools import get_math_tool

# Similarity threshold for considering document context
SIMILARITY_THRESHOLD = 0.7

def create_vector_store(chunks, embeddings, persist_directory=PERSIST_DIRECTORY):
    """Create a vector store from the document chunks."""
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vectorstore

def evaluate_context_template():
    """Template for context evaluation."""
    return """You are an expert at evaluating information relevance. Analyze if the following context is relevant and sufficient for answering the question.

    Question: {input}

    Retrieved Context:
    {context}

    Evaluate:
    1. Is this context relevant to the question? (Score 0-10)
    2. Is this context sufficient to answer the question completely? (Yes/No)
    3. Would general knowledge significantly improve the answer? (Yes/No)

    Provide your evaluation in this format:
    RELEVANCE_SCORE: [0-10]
    SUFFICIENT: [Yes/No]
    NEEDS_GENERAL: [Yes/No]
    """

def create_rag_chain(retriever, llm, memory=None):
    """Create an intelligent RAG chain that evaluates context relevance."""
    
    # Initialize math tool
    math_tool = get_math_tool()
    
    # Context evaluation prompt
    eval_prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=evaluate_context_template()
    )

    # Main conversation prompt
    template = """You are a helpful study assistant with strong mathematical capabilities. You can help with both general questions and mathematical problems.

    If the question involves mathematical calculations, equations, or plotting, use the Math Tool. Here's how to use it:
    1. For solving equations: Use operation "solve" with the equation (e.g., "x^2 + 2x + 1 = 0")
    2. For evaluating expressions: Use operation "evaluate" with the expression
    3. For plotting functions: Use operation "plot" with the function expression

    Previous conversation:
    {history}

    {context_note}
    
    Human: {input}

    Assistant: """

    prompt = PromptTemplate(
        input_variables=["context_note", "history", "input"],
        template=template
    )

    def get_context_with_scores(query):
        """Get relevant documents with their similarity scores."""
        if retriever is None:
            return []
            
        # Get documents using the retriever
        docs = retriever.get_relevant_documents(query)
        
        # Calculate similarity scores using the underlying vectorstore
        if hasattr(retriever, 'vectorstore'):
            embeddings = retriever.vectorstore._embedding_function
            query_embedding = embeddings.embed_query(query)
            doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]
            
            # Calculate cosine similarity
            scores = [np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)) 
                     for doc_emb in doc_embeddings]
            
            return list(zip(docs, scores))
        else:
            # If we can't get scores, assume a default high score for retrieved docs
            return [(doc, 0.8) for doc in docs]

    def evaluate_context(context, query):
        """Use LLM to evaluate context relevance and sufficiency."""
        eval_chain = LLMChain(llm=llm, prompt=eval_prompt)
        eval_result = eval_chain.predict(input=query, context=context)
        
        # Parse evaluation results
        lines = eval_result.split('\n')
        relevance = float(lines[0].split(':')[1].strip())
        sufficient = lines[1].split(':')[1].strip().lower() == 'yes'
        needs_general = lines[2].split(':')[1].strip().lower() == 'yes'
        
        return relevance, sufficient, needs_general

    class SmartChain(LLMChain):
        """Chain that intelligently decides how to use context and math capabilities."""
        
        def prep_inputs(self, inputs: dict) -> dict:
            query = inputs["input"]
            
            # Check if it's a math question
            if any(keyword in query.lower() for keyword in ["solve", "calculate", "evaluate", "plot", "equation", "graph"]):
                # Let the LLM use the math tool through the conversation template
                inputs["context_note"] = "This appears to be a math question. I'll help you solve it."
                return inputs
            
            # Regular document context handling
            docs_and_scores = get_context_with_scores(query)
            
            if not docs_and_scores:
                inputs["context_note"] = "Using my general knowledge to help you with this question."
                return inputs

            # Combine all relevant documents
            context = "\n\n".join(doc.page_content for doc, _ in docs_and_scores)
            avg_score = np.mean([score for _, score in docs_and_scores])
            
            # Evaluate context
            relevance, sufficient, needs_general = evaluate_context(context, query)
            
            if relevance < 5:  # Low relevance
                inputs["context_note"] = "Using my general knowledge as the document context wasn't relevant enough."
            elif sufficient and not needs_general:
                inputs["context_note"] = f"Based on your documents:\n{context}"
            else:
                inputs["context_note"] = f"Using both your documents and my general knowledge:\n{context}"
            
            return inputs

    # Create the smart chain with math capabilities
    chain = SmartChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    # Add math tool to the chain's allowed tools
    chain.tools = [math_tool]
    
    return chain