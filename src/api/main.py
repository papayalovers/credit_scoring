from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router
import uvicorn
from api.db import init_db
from api.router import DB_PATH 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # specified allow ip address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router, prefix="/api/v1")

if __name__ == '__main__':
    init_db(DB_PATH)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)