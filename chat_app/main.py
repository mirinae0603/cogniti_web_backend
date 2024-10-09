from pydantic import BaseModel
from fastapi import HTTPException, FastAPI, Response, Depends
from uuid import UUID, uuid4
import uvicorn
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from typing import List
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/create_session/{name}")
async def create_session(name: str, response: Response):
    session = uuid4()
    data = SessionData(username=name, conversation=[])  
    await backend.create(session, data)
    cookie.attach_to_response(response, session)
    return {"message": f"Session created for {name}. You can now start chatting."}


@app.post("/chat", dependencies=[Depends(cookie)])
async def chat(request: TextRequest, session_id: UUID = Depends(cookie), session_data: SessionData = Depends(verifier)):
    
    session_data.conversation.append(f"You: {request.text}")
    
    
    bot_response = f"Bot: I received your message: {request.text}"
    session_data.conversation.append(bot_response)
    
    
    await backend.update(session_id, session_data)
    
    return {"conversation": session_data.conversation}

@app.get("/whoami", dependencies=[Depends(cookie)])
async def whoami(session_data: SessionData = Depends(verifier)):
    return session_data


@app.post("/clear_session", dependencies=[Depends(cookie)])
async def clear_session(response: Response, session_id: UUID = Depends(cookie)):
    await backend.delete(session_id)
    cookie.delete_from_response(response)
    return {"message": "Session ended and conversation cleared. You can start a new session."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)