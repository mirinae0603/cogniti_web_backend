# cogniti_web_backend
Cogniticore Website Chat Backend

# FastAPI Session-based Chat Application

This project is a simple session-based chat application built with FastAPI. The app allows users to create sessions, exchange messages with a bot, and maintain session-based conversations.

## Features

- **Session management**: Each user gets a unique session, allowing for personalized conversations.
- **In-memory backend**: Sessions are stored in memory, which makes it suitable for lightweight or development environments.
- **Bot interaction**: Users can chat with a simple bot that echoes received messages.
- **Session persistence**: Conversations are preserved throughout the session until explicitly cleared.

## Technologies Used

- **[FastAPI](https://fastapi.tiangolo.com/)**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python-type hints.
- **[uvicorn](https://www.uvicorn.org/)**: A lightning-fast ASGI server for Python, used to serve FastAPI applications.
- **[pydantic](https://pydantic-docs.helpmanual.io/)**: Data validation and settings management using Python type annotations.
- **[fastapi-sessions](https://pypi.org/project/fastapi-sessions/)**: A session management library for FastAPI that allows user sessions with backends like in-memory or databases.

## Installation

To set up the project locally, follow these steps:

### Clone the Repository

```bash
git clone https://github.com/your-username/fastapi-chat-app.git
cd fastapi-chat-app
