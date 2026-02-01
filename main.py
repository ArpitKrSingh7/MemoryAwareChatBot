from dotenv import load_dotenv
import os
from openai import OpenAI
from mem0 import Memory

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url=os.getenv("GOOGLE_API_URL")
)

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4o"
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small"
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://c60f5694.databases.neo4j.io",
            "username": "neo4j",
            "password": "_3hjXXi82K8P59ayy5WPGNsTMtjNMb3usrpVbCmcRfw"
        }
    },
    "version": "v1.1",
}

m = Memory.from_config(config_dict=config)

MAX_LIMIT = 5
conversation_history = []

while(True):
    query = input("Enter your query: ")
    if(query.lower() in ["exit", "quit", "q"]):
        break
    
    result= m.add(query,user_id="user_2")
    # print("Response:", result)
    
    system_prompt = """
    
    You are an Memory aware AI tool , Which answers the user query based on the user memories.
    You have a knowledge of the user memories and you try to use the user memories to answer the query.
    Never say:
    1. it like "According to your memory" or "As per your memory" or "As you told me earlier"
    2. Given your love for movies, and knowing that "Conjuring the Last Rites" is a favorite,
    
    Rules :
    1. You will answer in a personalized manner according to the user memory whose access is with you.
    2. You will asnwer in discriptive manner. (not more than 100 words)
    3. You will always try to use the user memories to answer the query.
    4. If you don't have the relevant memory, Tell the user that you haven't told me about this yet.
    6. Use memory abstractly: give advice, suggestions, or examples without revealing personal details.
    7, Don't use the exact memory text, instead, use it to generate a relevant and personalized response.
    8. surprise the user with your personalized answers.
    
    Memory : {0}
    
    Examples :
    
    Input: Can I watch a movie tonight?
    Output: Absolutely! You could try watching Conjuring the Last Rites tonight — sounds like a thrilling choice!
    
    Input: Hello.
    Output: Hello , How can I help you?
    
    Input: I am Arpit.
    Output: Hello Arpit , How can I help you?
  
    Inout: I want to go outdoors this weekend.
    Output Great idea! You could check out some nearby hiking trails or nature spots for a refreshing day.
    
    Input: Who is my best friend?
    Output: Looks Like , you haven't shared this tea to me yet eheh.
    
    Input: Remind me about my hobbies.
    Output:  you enjoy hiking, photography, and cooking. Those sound like wonderful hobbies to have!
    
    Input: What is my favorite food?
    Output: You love pizza, sushi, and chocolate. Delicious choices!\
    
    Input: Do I like traveling?
    Output: Yes, you love exploring new places and experiencing different cultures. Traveling is one of your passions!  
    
    Input: which Movie , i shoud watch?
    Output: I Highly recommend you might "Inception" for a mind-bending thriller or "The Grand Budapest Hotel" for a quirky comedy.
    
    """
    relevant_memory = m.search(query, user_id="user_2")
    system_prompt = system_prompt.format(relevant_memory)
    
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for q, a in conversation_history[-MAX_LIMIT:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        temperature=0.7,
        messages=messages
    )
    print(f"AI said : {response.choices[0].message.content}")
    conversation_history.append((query, response.choices[0].message.content))
    