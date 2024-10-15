from pydantic import BaseModel
from fastapi import HTTPException, FastAPI, Response, Depends
from uuid import UUID, uuid4
import uvicorn, openai
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import requests,json
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from threading import Thread
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime, timezone

# Constants
OPENROUTER_API_KEY = "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"  # Replace this with your OpenRouter API Key

OPENROUTER_MODEL = "nousresearch/hermes-3-llama-3.1-405b:free"

# Secure connect bundle for DataStax Astra
cloud_config = {
    'secure_connect_bundle': 'secure-connect-cogniticore-chat-records.zip'
}

# Token JSON file for authentication
with open("shivanshdarshan@gmail.com-token.json") as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]

# Create authentication provider and connect to the cluster
auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

def insert_chat_pair(question, answer, model_used, summary_used, session_id="experiment"):
    created_at = str(datetime.now(timezone.utc))
    session.execute("""
    INSERT INTO chats.qa_pairs (id, session_id, question, answer, created_at, model_used, summary_used)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (uuid4(), session_id, question, answer, created_at, model_used, summary_used))

class SessionData(BaseModel):
    username: str
    raw_chat_data: List[str] = []
    summary: str
    conversation: List[str] = []

class TextRequest(BaseModel):
    text: str
    session_id: UUID  # Now session_id is passed in the request body

cookie_params = CookieParameters(max_age=3600, httponly=False, secure=False, samesite="none")

# Initialize session cookie
cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",  # Replace this with a secure secret key in production
    cookie_params=cookie_params,
)

backend = InMemoryBackend[UUID, SessionData]()
active_sessions: List[UUID] = []

class StreamRequest(BaseModel):
    user_message: str
    session_id: str

class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(
        self,
        *,
        identifier: str,
        auto_error: bool,
        backend: InMemoryBackend[UUID, SessionData],
        auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        """If the session exists, it is valid"""
        return True

verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)

app = FastAPI()

allowed_origins = ["https://www.cogniticore.com", "https://www-cogniticore-com.filesusr.com", "https://pgvjqr.csb.app", "https://pgvjqr.csb.app/", "http://localhost:8000/prod.html", "http://localhost:8000/", "http://localhost:8000"]

# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allow your specific frontend app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a new session
@app.post("/create_session/{name}")
async def create_session(name: str, response: Response):
    session = uuid4()  # Generate new session ID
    data = SessionData(username=name, raw_chat_data=[], summary=f"", conversation=[])
    await backend.create(session, data)
    cookie.attach_to_response(response, session)  # Attach cookie with session ID
    active_sessions.append(session)
    return {"message": f"Session created for {name}. You can now start chatting.", "session_id": session}

async def sumarize_conversation(session_id):
    print("Summarization Initiated")
    # Get the conversation from the session
    session_data = await backend.read(UUID(session_id))
    # Summarize the conversation
    # if len(session_data.raw_chat_data)%5==0 and len(session_data.raw_chat_data)>5:
    data_to_summarize = ""
    if len(session_data.raw_chat_data)>5:
        data_to_summarize = ''.join(session_data.raw_chat_data)[-5:]
    else:
        data_to_summarize = ''.join(session_data.raw_chat_data)[:]
    response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
                    },
                    data=json.dumps({
                        "model": "meta-llama/llama-3-8b-instruct:free", 
                        "messages": [
                        {
                            "role": "user",
                            "content": f"{data_to_summarize} Please provide a concise summary of the following conversations, user intent and answers provided. ensure correct data in summary no vague stuffs and capture asmuch fine details as possible capture relevant keywords about user and other details which can acta as reference answer in 3-4 lines."
                        }
                        ]
                        
                    })
                    )
    print("Summary Generated : ", response.json())
    try:
        session_data.summary = response.json()['choices'][0]['message']['content']
        await backend.update(UUID(session_id), session_data)
    except Exception as e:
        try:
            response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
                    },
                    data=json.dumps({
                        "model": "meta-llama/llama-3.1-70b-instruct:free", 
                        "messages": [
                        {
                            "role": "user",
                            "content": f"""{data_to_summarize} Please provide a concise summary of the following conversations, user intent and answers provided.  ensure correct data in summary no vague stuffs and capture asmuch fine details as possible.
                                        Capture keypoints and details to know everything about the conversation"""
                        }
                        ]
                        
                    })
                    )
            print("Summary Generated : ", response.json())
            session_data.summary = response.json()['choices'][0]['message']['content']
            await backend.update(UUID(session_id), session_data)
        except:
            pass
        
        # session_data.summary = f"Summary Generation Failed - Reason : {e}"
        # await backend.update(UUID(session_id), session_data)
        pass

async def azure_sumarize_conversation(session_id):
    print("Summarization Initiated")
    # Get the conversation from the session
    session_data = await backend.read(UUID(session_id))
    # Summarize the conversation
    # if len(session_data.raw_chat_data)%5==0 and len(session_data.raw_chat_data)>5:
    data_to_summarize = ""
    if len(session_data.raw_chat_data)>11:
        data_to_summarize = ''.join(session_data.raw_chat_data)[-10:]
    else:
        data_to_summarize = ''.join(session_data.raw_chat_data)[:]

    payload = {
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": f"""{data_to_summarize} Please provide a concise summary of the following conversations, user intent and answers provided.  ensure correct data in summary no vague stuffs and capture asmuch fine details as possible.
                                        Capture keypoints and details to know everything about the conversation"""
            }
        ]
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800,
    # "stream":True
    }

    ENDPOINT = api_base

    print(payload)
    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        print(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(e)
        pass
    try:
        session_data.summary = response.json()['choices'][0]['message']['content']
        await backend.update(UUID(session_id), session_data)
    except Exception as e:
        print(e)
        pass

async def generate_response(user_message: str, session_id: str):
    # Prepare the data to send to OpenRouter API

    session_data = await backend.read(UUID(session_id))

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"""
                Home Page URL - https://www.cogniticore.com
                Case Studies NLP URL - https://www.cogniticore.com/nlp
                Case Studies CV URL - https://www.cogniticore.com/cv
                Capabilities URL - https://www.cogniticore.com/capabilities
                About Us URL     - https://www.cogniticore.com/aboutus
                Contact Us URL   - https://www.cogniticore.com/contactus

                Enable AI Dominance in Your Industry

                AI Chat: Fast, Personal, Evolving  

                Transform customer engagement with an industry-specific chatbot that automates support, provides insightful analytics, and delivers seamless lead classification—minimizing manual effort.

                Tailored AI Solutions  

                - Natural Language Processing (NLP) & Large Language Models (LLMs): Transform text into actionable insights with advanced NLP and LLM capabilities, enabling sentiment analysis, summarization, and effective communication for data-driven decisions.

                - Computer Vision: Harness visual data to automate processes and unlock insights, from object detection to 3D reconstruction, enhancing operational efficiency.

                - MLOps: Optimize your AI lifecycle with MLOps solutions, ensuring integration, monitoring, and scalability for thriving AI projects.

                Capabilities: Harnessing AI for Efficiency  

                - NLP: Our chatbots utilize advanced NLP techniques for seamless customer interactions, delivering instant support and personalized experiences.

                - LLMs: Revolutionize technology interaction with advanced text generation and conversation capabilities, automating tasks and enhancing customer engagement.

                - Workflow Automation: Automate repetitive tasks to reduce errors and streamline operations, allowing teams to focus on valuable work.

                - MLOps (CLONE): CLONE (Continuous Learning Operations & Neural Efficiency) optimizes ML workflows, enhancing model deployment, monitoring, and management for better decision-making.

                - Computer Vision: Enable machines to interpret visual data, leveraging algorithms for defect detection and actionable insights across industries.

                - TryOn Technology: Revolutionize shopping with virtual try-ons, transforming business processes through enhanced automation and decision-making.

                About Us: The Power of Communication  

                - Vision: Pioneer AI’s future with transformative solutions that empower partners to lead their industries.
                - Mission: Empower partners to achieve dominance through innovative AI solutions, unlocking potential and driving growth.

                Our Values: A.C.T.I.O.N.  

                - Accountability
                - Customer-Centric  
                - Trust & Transparency 
                - Innovation-Driven 
                - Operational Excellence
                - Nimbleness


                Prev - Conversation Summary : {session_data.summary} 
                For the User asked Question {user_message} answer it on these guidelines
                1. Answers should be always in points with fine details.
                2. If question is not related to above context by more than 70% only Give `Please ask relevant question` as answer
                3. Add relevant URL to answer as well for Reference
                4. Answer as if you own the information
                5. Only answer to the Point no useless Information
                6. Answers must not be too big around 100 words
                7. If Question is greeting answer it professionally as you are assistant guiding clients for our company Cogniticore"""
            },
        ],
        "stream": True  # Enable streaming response
    }

    # Send request to OpenRouter API
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        data=json.dumps(payload),
        stream=True
    )

    # Process the response as it comes in
    

    # Process the response as it comes in
    combined_text = ""
    # for chunk in response.iter_content(chunk_size=None):
    #     if chunk:
    #         decoded_chunk = chunk.decode()
    #         # Split the chunks based on the expected format
    #         parts = decoded_chunk.split("data: ")
    #         for part in parts[1:]:  # Skip the first part which is before the first 'data: '
    #             # Handle the 'null' case in the string safely
    #             if part.strip() == 'null':
    #                 continue
                
    #             # Safely load the JSON data
    #             try:
    #                 data_chunk = json.loads(part)
    #                 # Check if 'choices' and 'delta' exist and extract content
    #                 content = data_chunk['choices'][0]['delta'].get('content', '')
                    
    #                 # Append content to combined_text and yield the content for streaming
    #                 combined_text += content
    #                 # content = "Yielding for streaming response"
    #                 # print(content)
    #                 yield content + "\n"  # Yielding for streaming response

    #             except (json.JSONDecodeError, KeyError) as e:
    #                 # Handle any JSON errors or missing keys gracefully
    #                 print(f"Error processing chunk: {e}")

    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            if "id" in chunk.decode():
                targ = "None".join(chunk.decode().split("null")).split("data: ")
                for kp in targ[1:]:
                    kp = kp.split(": OPENROUTER PROCESSING")[0]
                    if "id" in kp and "content" in kp and "delta" in kp and "choices" in kp:
                        combined_text += eval(kp)['choices'][0]['delta']['content']
                        yield eval(kp)['choices'][0]['delta']['content']

    session_data = await backend.read(UUID(session_id))
    insert_chat_pair(user_message, combined_text, "hermes-3-llama-405b", session_data.summary, session_id)
    if session_data:
        session_data.conversation.append(f"bot: {combined_text}")
        session_data.raw_chat_data.append(f"ques: {user_message} ans: {combined_text}")
        await backend.update(UUID(session_id), session_data)

    print("Initiatning Summary Generation")
    await sumarize_conversation(session_id)
    

    # Return the generated response
     # Stream the response

@app.post("/stream")
async def chat(request: StreamRequest):
    user_message = request.user_message
    session_id_str = request.session_id

    try:
        # Convert session_id_str to UUID type
        session_id = UUID(session_id_str)
    except ValueError:
        # Return error if the session ID format is invalid
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    # Check if session exists in the backend
    session_data = await backend.read(session_id)
    if session_data is None:
        # Return an error if session ID does not exist
        raise HTTPException(status_code=404, detail="Please refresh your website your session ended due to inactivity")

    # Append the user's message to the session conversation
    session_data.conversation.append({"user": user_message})
    await backend.update(session_id, session_data)

    # Return a StreamingResponse that generates the bot's response
    return StreamingResponse(generate_response(user_message, str(session_id)), media_type="text/plain")



#=============================== OPENAI STREAMING ==============================================

AZURE_OPENAI_API_KEY = "ee400dfb08854a3196e0d7e1daa924fc" 
AZURE_OPENAI_ENDPOINT = "https://ccw.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"




    # for chunk in response.iter_lines():
    #     print("Phase 0: ", chunk)
    #     decoded_chunk = chunk.decode()
    #     print("Phase : 1", decoded_chunk)
        # if len(decoded_chunk)>0:
        #     decoded_chunk= eval("True".join("False".join("None".join(decoded_chunk.split("data: ")[-1].split("null")).split("false")).split("true")))
        #     print("Phase : 2", decoded_chunk)
        #     if len(decoded_chunk["choices"])>0:
        #         try:
        #             # Decode each chunk and handle the streaming response
                    
        #             # decoded_chunk = json.loads(chunk.decode('utf-8'))
                    
        #             print("Result : ", decoded_chunk)

        #             chunk_message = decoded_chunk['choices'][0]['delta'].get('content', '')
        #             if chunk_message:
        #                 combined_text += chunk_message
        #                 yield chunk_message  # Stream each chunk of content
        #         except (KeyError, json.JSONDecodeError):
        #             continue

    # Store the final combined response in the session log
    # if session_id not in sessions:
    #     sessions[session_id] = {"messages": []}  # Example session data

    # sessions[session_id]["messages"].append({"bot": combined_text})

from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = AZURE_OPENAI_ENDPOINT, 
  api_key=AZURE_OPENAI_API_KEY,  
  api_version="2024-07-18"
)

async def generate_response_azure(user_message: str, session_id: str):
    # Prepare the data to send to OpenRouter API
    session_data = await backend.read(UUID(session_id))
    datad = f"""
                Home Page URL - https://www.cogniticore.com
                Case Studies NLP URL - "https://www.cogniticore.com/nlp
                Case Studies CV URL - https://www.cogniticore.com/cv
                Capabilities URL - https://www.cogniticore.com/capabilities
                About Us URL     - https://www.cogniticore.com/aboutus
                Contact Us URL   - https://www.cogniticore.com/contactus

                Enable AI Dominance in Your Industry

                AI Chat: Fast, Personal, Evolving  

                Transform customer engagement with an industry-specific chatbot that automates support, provides insightful analytics, and delivers seamless lead classification—minimizing manual effort.

                Tailored AI Solutions  

                - Natural Language Processing (NLP) & Large Language Models (LLMs): Transform text into actionable insights with advanced NLP and LLM capabilities, enabling sentiment analysis, summarization, and effective communication for data-driven decisions.

                - Computer Vision: Harness visual data to automate processes and unlock insights, from object detection to 3D reconstruction, enhancing operational efficiency.

                - MLOps: Optimize your AI lifecycle with MLOps solutions, ensuring integration, monitoring, and scalability for thriving AI projects.

                Capabilities: Harnessing AI for Efficiency  

                - NLP: Our chatbots utilize advanced NLP techniques for seamless customer interactions, delivering instant support and personalized experiences.

                - LLMs: Revolutionize technology interaction with advanced text generation and conversation capabilities, automating tasks and enhancing customer engagement.

                - Workflow Automation: Automate repetitive tasks to reduce errors and streamline operations, allowing teams to focus on valuable work.

                - MLOps (CLONE): CLONE (Continuous Learning Operations & Neural Efficiency) optimizes ML workflows, enhancing model deployment, monitoring, and management for better decision-making.

                - Computer Vision: Enable machines to interpret visual data, leveraging algorithms for defect detection and actionable insights across industries.

                - TryOn Technology: Revolutionize shopping with virtual try-ons, transforming business processes through enhanced automation and decision-making.

                About Us: The Power of Communication  

                - Vision: Pioneer AI’s future with transformative solutions that empower partners to lead their industries.
                - Mission: Empower partners to achieve dominance through innovative AI solutions, unlocking potential and driving growth.

                Our Values: A.C.T.I.O.N.  

                - Accountability
                - Customer-Centric  
                - Trust & Transparency 
                - Innovation-Driven 
                - Operational Excellence
                - Nimbleness

                Contact Email Address - connect@cogniticore.com
                Contact Phone Number  - Tel: +1 (302) 343-3422
                Contact Address - 16192 Coastal Highway, Lewes Delaware 19958, County of Sussex
                Linkdin URL - https://www.linkedin.com/company/cogniti-core
                Instagram URL - https://www.instagram.com/COGNITICORE/
                Thread(Twitter) URL -  https://x.com/CognitiCore

                Prev - Conversation Summary : {session_data.summary}  for reference
                For the User asked Question {user_message} answer it on these guidelines
                1. Answers should be always in points with fine details
                3. Add relevant URL to answer as well.
                4. Answer as if you are an assistant of Cogniticore and represeting it
                5. Only answer to the Point no useless and vague Information should be spurted out 
                6. Answers must not be too big around 100 words
                7. Remove brackets and other text around provided url's in the answer
                8. If {user_message} is irrelevant to cbove context answer Please ask relevant questions
                9. If question is related to Conversationaly Summary answer it properly with fine details"""
    
    # datad = f"{user_message} answer in 3 points 50 words and don't answer in markdown format only text format"
    payload = {
                "messages": [
                    {
                    "role": "system",
                    "content": [
                        {
                        "type": "text",
                        "text": datad,
                        }
                    ]
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 800,
                "stream":True
                }
    
    headers = {
                    "Content-Type": "application/json",
                    "api-key": AZURE_OPENAI_API_KEY,
                }
    # Send request to OpenRouter API
    combined_text = ""
    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    for line in response.iter_lines():
        if line:
            answer = line.decode().split("data: ")[-1]
            # print(answer)
            if "delta" in answer:
                data = "None".join(answer.split("null"))
                data = "False".join(data.split("false"))
                data = eval(data)
                # print("Orig : ", data)
                if "content"in data["choices"][0]['delta']:
                  print(data["choices"][0]['delta']['content'], end="")
                  result_to_stream = data["choices"][0]['delta']['content']
                  result_to_stream = "".join(result_to_stream.split("**"))
                  result_to_stream = " ".join(result_to_stream.split("["))
                  result_to_stream = " ".join(result_to_stream.split("]"))
                  combined_text+=result_to_stream
                  yield  result_to_stream
    

    # After processing all chunks, append bot response to the session conversation
    session_data = await backend.read(UUID(session_id))
    insert_chat_pair(user_message, combined_text, "gpt-4o-mini", session_data.summary, session_id)
    if session_data:
        session_data.conversation.append(f"bot: {combined_text}")
        session_data.raw_chat_data.append(f"ques: {user_message} ans: {combined_text}")
        await backend.update(UUID(session_id), session_data)

    print("Initiatning Summary Generation")
    await azure_sumarize_conversation(session_id)

@app.post("/azure_stream")
async def chat(request: StreamRequest):
    user_message = request.user_message
    session_id_str = request.session_id

    try:
        # Convert session_id_str to UUID type
        session_id = UUID(session_id_str)
    except ValueError:
        # Return error if the session ID format is invalid
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    # Check if session exists in the backend
    session_data = await backend.read(session_id)
    if session_data is None:
        # Return an error if session ID does not exist
        raise HTTPException(status_code=404, detail="Session not found. Please create a new session.")

    # Append the user's message to the session conversation
    session_data.conversation.append({"user": user_message})
    await backend.update(session_id, session_data)

    # Return a StreamingResponse that generates the bot's response
    return StreamingResponse(generate_response_azure(user_message, str(session_id)), media_type="text/plain")

#== ====================================================================

# Get user information using the session ID
@app.get("/whoami")
async def whoami(session_id: UUID):
    session_data = await backend.read(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_data

@app.post("/update_model")
async def update_model(model_name: str):
    global OPENROUTER_MODEL
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name cannot be empty")
    OPENROUTER_MODEL = model_name
    return {"message": f"Model updated to {model_name}"}

@app.get("/current_model")
async def get_current_model():
    return {"current_model": OPENROUTER_MODEL}

# Get the total number of active sessions
@app.get("/session_counts")
async def session_counts():
    return {"session_counts": len(active_sessions)}

# Get the list of current session IDs
@app.get("/current_session_ids")
async def current_session_ids():
    all_sessions = {}
    for session_id in active_sessions:
        session_data = await backend.read(session_id)
        if session_data is not None:  # Ensure session data exists
            all_sessions[session_id] = session_data
    return {"current_session_ids": all_sessions}

# Clear a specific session by session ID
@app.post("/clear_session")
async def clear_session(request: TextRequest):
    session_id = request.session_id
    await backend.delete(session_id)
    active_sessions.remove(session_id)  # Remove from active session list
    return {"message": "Session ended and conversation cleared. You can start a new session."}

# Clear all sessions
@app.post("/clear_all_sessions")
async def clear_all_sessions():
    for session_id in active_sessions:
        await backend.delete(session_id)
    active_sessions.clear()
    return {"message": "All sessions have been cleared."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501, timeout_keep_alive=60)
