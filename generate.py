from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from llm import llm

def generate(query, search_results) -> str:
    context = "\n\n".join([result.payload['content'] for result in search_results])
    system_message = SystemMessagePromptTemplate.from_template("""
    # Role and Objective
    You will act as a smart AI chatbot that answers questions by citing from the provided information.
    
    # Instructions
    Your rules for answering:
    - Respond in the language in which the user question has been asked.
    - You will be given context that comes from various sources of information.
    
    # Steps
    - Decide which information within the provided context is relevant to answer the question.
    - Revise your information list and only keep information that contains parts of your answer.
    - If the provided information within the context does not contain the answer: Let me know. 
    
    # Output Format
    - Always use Markdown as the output format for your entire answer.
    - If you refer to sources of information within your answer, please use the references if provided.
    
    # Information:
    {context}
    """)
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": query})
    return answer.content
