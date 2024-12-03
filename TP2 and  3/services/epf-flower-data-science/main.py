import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.app import get_application
from src.api.routes.data import router as data_router

app = get_application()

# Step 3: Redirect root ("/") to the Swagger UI automatically
@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

app.include_router(data_router) 

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)