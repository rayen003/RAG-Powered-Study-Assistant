from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from app.templates import TEMPLATES

def general_workflow(memory, question):
    llm = ChatOpenAI()  # No need to pass api_key here

    chain = (
        {
            "chat_history": lambda _: memory.load_memory_variables({}).get("chat_history", ""),
            "question": RunnablePassthrough()
        }
        | TEMPLATES["general"]
        | llm
    )

    # Execute the chain and get the response
    response = chain.invoke(question)

    # Format the final output
    output = {
        "answer": response.content if hasattr(response, 'content') else str(response),
        "chat_history": memory.load_memory_variables({}).get("chat_history", "")
    }

    return output

# Example test usage
if __name__ == "__main__":
    # Create test memory
    test_memory = ConversationSummaryMemory(
        llm=ChatOpenAI(),
        memory_key="chat_history",
        input_key="question", 
        output_key="answer",
        return_messages=True
    )
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "Can you explain neural networks?",
        "How does backpropagation work?"
    ]
    
    # Run workflow with test questions
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = general_workflow(test_memory, question)
        
        # Print only the answer
        print(f"Response: {response['answer']}")
        
        # Save the interaction to memory
        test_memory.save_context(
            {"question": question},
            {"answer": response['answer']}
        )
