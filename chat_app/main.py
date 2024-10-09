from pydantic import BaseModel
from fastapi import HTTPException, FastAPI, Response, Depends
from uuid import UUID, uuid4
import uvicorn
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import requests
import json

# Constants
OPENROUTER_API_KEY = "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"


class SessionData(BaseModel):
    username: str
    conversation: List[str] = []  


class TextRequest(BaseModel):
    text: str


cookie_params = CookieParameters()


cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",  
    cookie_params=cookie_params,
)


backend = InMemoryBackend[UUID, SessionData]()

active_sessions: List[UUID] = []

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

app = FastAPI()

verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pgvjqr.csb.app"],  # Allow your specific frontend app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/create_session/{name}")
async def create_session(name: str, response: Response):
    session = uuid4()
    data = SessionData(username=name, conversation=[])  
    await backend.create(session, data)
    cookie.attach_to_response(response, session)
    active_sessions.append(session) 
    print("Headers : ", response.headers)
    return {"message": f"Session created for {name}. You can now start chatting."}


@app.post("/chat2", dependencies=[Depends(cookie)])
async def chat(request: TextRequest, session_id: UUID = Depends(cookie), session_data: SessionData = Depends(verifier)):
    
    session_data.conversation.append(f"You: {request.text}")
    
    
    bot_response = f"Bot: I received your message: {request.text}"
    session_data.conversation.append(bot_response)
    
    
    await backend.update(session_id, session_data)
    
    return {"conversation": bot_response}

@app.post("/chat", dependencies=[Depends(cookie)])
async def chat(request: TextRequest, session_id: UUID = Depends(cookie), session_data: SessionData = Depends(verifier)):
    
    # Append user's message to the conversation
    session_data.conversation.append(f"You: {request.text}")
    
    # Prepare the data to send to OpenRouter API
    payload = {
        "model": "liquid/lfm-40b:free", 
        "messages": [
            {
                "role": "user",
                "content": f"""
                Case Studies URL - "https://www.cogniticore.com/nlp"
                Capabilities URL - "https://www.cogniticore.com/capabilities"
                About Us URL     - "https://www.cogniticore.com/aboutus"
                Contact Us URL   - "https://www.cogniticore.com/contactus" 
                Enabling AI Dominance in Your Industry for You AI Chat: Fast, Personal, Evolving Transform customer engagement with an industry-specific chatbot that automates support, provides insightful analytics, and delivers seamless lead classification—all while minimizing manual effort AI Solutions Tailored for Your Needs Natural Language Processing & Large Language Models: - Transform text into actionable insights with our advanced NLP and LLM capabilities. Whether it’s sentiment analysis, summarization, or chatbots, we help you communicate effectively and make data- driven decisions Computer Vision: - Harness the power of visual data to unlock insights and automate processes. From object detection to 3D reconstruction, our solutions enhance operational efficiency and drive innovation. MLOps:- Optimize your AI lifecycle with our MLOps solutions. We ensure seamless integration, monitoring, and scalability, allowing your AI projects to thrive in dynamic environments.Capabilities: - Harnessing AI for Efficiency Natural Language Processing: - View our Case Studies Natural Language Processing (NLP) empowers our chatbots to understand and engage in human language seamlessly. By utilizing advanced NLP techniques, we enhance customer interactions, delivering instant support and personalized experiences. Transform your communication and drive efficiency across industries with our innovative chatbot solutions powered by NLP. LLM – DOC: - View our Case Studies Large Language Models (LLMs) revolutionize the way we interact with technology. Our LLM solutions provide advanced text generation, understanding, and conversation capabilities, enabling businesses to automate tasks and enhance customer engagement. Experience the power of LLMs to drive innovation and improve efficiency across diverse applications in your industry.Workflow Automation: - View our Case Studies Workflow automation enhances efficiency by automating repetitive tasks and reducing errors. Our solutions integrate smoothly with your existing systems, allowing teams to focus on more valuable work. Streamline operations and boost productivity with tailored automation strategies designed to drive growth in your organization. MLOps – CLONE: - View our Case Studies CLONE (Continuous Learning Operations & Neural Efficiency) revolutionizes MLOps by optimizing machine learning workflows. Our services streamline model deployment, monitoring, and management, ensuring seamless integration and reliable performance, empowering your business to leverage AI for enhanced decision-making and innovation. Computer Vision: - View our Case Studies Computer vision enables machines to interpret and understand visual data. Our computer vision services, including defect detection, leverage advanced algorithms to analyze images and videos, providing actionable insights that enhance decision-making, improve automation, and drive innovation across various industries for your business.Try- ON: - View our Case Studies TryOn technology revolutionizes shopping with virtual try-ons. Our computer vision services use advanced algorithms to analyze images and videos, delivering actionable insights for defect detection, enhanced automation, and improved decision-making, ultimately transforming your business processes across various industries. About Us: We Believe in The Power of Communication Vision: - To pioneer the future of AI by creating transformative solutions that empower our partners to lead their industries, shaping a world where innovation and intelligence drive sustained success. Mission: - Empowering our partners to achieve industry dominance through innovative AI solutions, unlocking their full potential and driving transformative growth.Our Values: A.C.T.I.O.N. • Accountability: We are direct and transparent, fostering trust through no- nonsense communication and honest dealings. • Customer-Centric: Our client's success is our priority, and their needs come first in everything we do. • Trust & Transparency: We build strong, trusted relationships through clear communication and ethical actions. • Innovation-Driven Our adaptable, future-proof solutions are designed to evolve and keep you ahead of the curve. • Operational Excellence We follow strong, structured processes to deliver high- quality, resilient outcomes. • Nimbleness We constantly improvise, adapt, and overcome challenges to deliver exceptional results.
                For the User asked Question {request.text} answer it on these guidelines
                1. Answers should look presentable and easy to understand
                2. If question is not related to above context by more than 70% only Give `Please ask relevant question` as answer
                3. Add relevant URL to answer as well for Reference
                4. Answer as if you own the information
                5. Only answer to the Point no useless Information
                6. Answers must not be too big"""
            },
        ],
        "stream": True  # Enable streaming response
    }
    
    # Send request to OpenRouter API
    # response = requests.post(
    #     url="https://openrouter.ai/api/v1/chat/completions",
    #     headers={
    #         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    #     },
    #     data=json.dumps(payload),
    #     stream=True  # Streaming is enabled for the request as well
    # )
    
    # # Process the response as it comes in
    # combined_text = ""

    # for chunk in response.iter_content(chunk_size=None):
    #     if chunk:
    #         print(chunk.decode())
    #         if "id" in chunk.decode():
    #             targ = "None".join(chunk.decode().split("null")).split("data: ")
    #             for kp in targ[1:]:
    #                 kp = kp.split(": OPENROUTER PROCESSING")[0]
    #                 if "id" in kp and "content" in kp and "delta" in kp and "choices" in kp:
    #                     combined_text += eval(kp)['choices'][0]['delta']['content']
    
    # # Append bot response to the conversation
    # session_data.conversation.append(f"Bot: {combined_text}")
    
    # # Update session data in the backend
    # await backend.update(session_id, session_data)
    combined_text = f"AI Models Disabled : Your Prompt{request.text}"
    
    return {"conversation": combined_text}

@app.get("/whoami", dependencies=[Depends(cookie)])
async def whoami(session_data: SessionData = Depends(verifier)):
    return session_data

@app.get("/session_counts")
async def session_counts():
    return {"session_counts": len(active_sessions)}

@app.get("/current_session_ids")
async def current_session_ids():
    return {"current_session_ids": active_sessions}

@app.post("/clear_session", dependencies=[Depends(cookie)])
async def clear_session(response: Response, session_id: UUID = Depends(cookie)):
    await backend.delete(session_id)
    cookie.delete_from_response(response)
    return {"message": "Session ended and conversation cleared. You can start a new session."}

@app.post("/clear_all_sessions")
async def clear_all_sessions(response: Response):
    for session_id in active_sessions:
        await backend.delete(str(session_id))  # Delete each session from the backend
    active_sessions.clear()  # Clear the active sessions list
    return {"message": "All sessions have been cleared."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
