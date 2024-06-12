from fastapi import FastAPI

from news_searcher import NewsSearcher

app = FastAPI()

# Create a searcher instance
neural_searcher = NewsSearcher(collection_name="articles_fr_newsapi")


@app.get("/api/search")
def search_startup(q: str):
    return {"result": neural_searcher.search(text=q)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1234)
