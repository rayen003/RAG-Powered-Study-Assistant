from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.config import MODEL_NAME
from typing import List, Optional, Union, BinaryIO
from langchain_core.output_parsers.string import StrOutputParser
from pydantic import BaseModel, ValidationError
import importlib
from app.templates import TEMPLATES
from app.memory_manager import MemoryManager
import os
import mimetypes

# Define a mapping of categories to workflows with absolute imports
WORKFLOW_MAP = {
    "General Processing": "app.workflows.general_workflow",
    "Advanced Processing": "app.workflows.advanced_workflow",
    "Basic Processing": "app.workflows.basic_workflow",
    "Data Analysis": "app.workflows.data_analysis_workflow",
    "Content Generation": "app.workflows.content_generation_workflow",
    "Multimodal Processing": "app.workflows.multimodal_workflow"
}

# Define a Pydantic model for the output
class CategoryOutput(BaseModel):
    category: str

class FileInput(BaseModel):
    """Represents a file input with its content and metadata"""
    content: bytes
    mime_type: str
    filename: str

def process_file(file_content: bytes, filename: str) -> Optional[FileInput]:
    """
    Process file content and return FileInput if valid.
    """
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    
    if mime_type is None:
        return None
        
    # List of supported MIME types
    supported_types = [
        'image/',           # All image types
        'application/pdf',  # PDF files
        'text/',           # Text files
        'application/msword',  # DOC files
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'  # DOCX files
    ]
    
    if not any(mime_type.startswith(supported) for supported in supported_types):
        return None
    
    return FileInput(
        content=file_content,
        mime_type=mime_type,
        filename=filename
    )

def input_classifier(model, initial_query: str, categories: List[str]):
    if not isinstance(initial_query, str):
        raise ValueError("initial_query must be a string.")
    if not isinstance(categories, list) or not all(isinstance(cat, str) for cat in categories):
        raise ValueError("categories must be a list of strings.")

    classification_chain = (
        {
            "query": RunnablePassthrough(),
            "categories": lambda x: ", ".join(x["categories"])
        }
        | TEMPLATES["classifier"]
        | model
    )

    selected_category = classification_chain.invoke({"query": initial_query, "categories": categories})

    if isinstance(selected_category, dict) and 'content' in selected_category:
        selected_category = selected_category['content'].strip()

    try:
        category_output = CategoryOutput(category=selected_category)
    except ValidationError:
        category_output = CategoryOutput(category="General Processing")

    if category_output.category not in WORKFLOW_MAP:
        category_output.category = "General Processing"

    return category_output.category

def is_file_input(file_path: str) -> bool:
    """
    Check if the input is an actual file and is of a supported type.
    """
    if not os.path.exists(file_path):
        return False
        
    # Get the MIME type of the file
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type is None:
        return False
        
    # List of supported MIME types
    supported_types = [
        'image/',           # All image types
        'application/pdf',  # PDF files
        'text/',           # Text files
        'application/msword',  # DOC files
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'  # DOCX files
    ]
    
    return any(mime_type.startswith(supported) for supported in supported_types)

def execute_workflow(category: str, memory, question: str, file_input: Optional[FileInput] = None):
    """
    Execute the appropriate workflow based on input type and category.
    """
    # If there's a valid file input, use multimodal workflow
    if file_input:
        workflow_module = "app.workflows.multimodal_workflow"
    else:
        workflow_module = WORKFLOW_MAP.get(category, "app.workflows.general_workflow")
    
    try:
        # Import the module using absolute import
        module = importlib.import_module(workflow_module)
        workflow_name = workflow_module.split('.')[-1]
        workflow_func = getattr(module, workflow_name)
        
        print(f"Debug: Executing workflow {workflow_module}")
        
        # Pass file_input to multimodal workflow if it exists
        if workflow_module == "app.workflows.multimodal_workflow" and file_input:
            return workflow_func(memory, question, file_input)
        return workflow_func(memory, question)
        
    except Exception as e:
        print(f"Error in workflow {workflow_module}: {str(e)}")
        if category != "General Processing":
            print("Falling back to general workflow")
            general_module = importlib.import_module("app.workflows.general_workflow")
            return general_module.general_workflow(memory, question)
        else:
            raise

def main():
    """
    Temporary CLI interface - will be replaced with proper UI later.
    """
    model = ChatOpenAI()
    memory = MemoryManager.get_memory()

    print("Welcome to the AI Assistant! Type 'exit' to end the conversation.")
    print("To analyze files, start your message with 'file:' followed by the file path.")
    
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        
        file_input = None
        selected_category = "General Processing"  # default category
        
        if query.startswith('file:'):
            filepath = query[5:].strip()
            try:
                with open(filepath, 'rb') as f:
                    file_input = process_file(f.read(), filepath)
                    if file_input:
                        selected_category = "Multimodal Processing"  # Force multimodal for files
                        query = f"Please analyze this file: {filepath}"
                    else:
                        print("Error: Unsupported file type.")
                        print("Supported files: images, PDFs, text files, DOC/DOCX files")
                        continue
            except FileNotFoundError:
                print(f"Error: File not found: {filepath}")
                continue
        else:
            # Only use classifier for non-file inputs
            selected_category = input_classifier(model, query, list(WORKFLOW_MAP.keys()))
            
        print(f"Selected workflow: {selected_category}")
        response = execute_workflow(selected_category, memory, query, file_input)
        print(f"Workflow response: {response['answer']}")

if __name__ == "__main__":
    main()


