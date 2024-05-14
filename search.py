import concurrent.futures
import json
import os
import re
import sqlite3
import threading
import traceback
from typing import List, Generator, Optional

import httpx
import openai
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

app = FastAPI()

# Constant values for the RAG model.
BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_MKT = "en-US"
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
SEARCHAPI_SEARCH_ENDPOINT = "https://www.searchapi.io/api/v1/search"

REFERENCE_COUNT = 8
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5

_default_query = "Who said 'live long and prosper'?"

_rag_query_text = """
You are a large language AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

stop_words = [
    "",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

_more_questions_prompt = """
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:
"""

def search_with_bing(query: str, subscription_key: str):
    params = {"q": query, "mkt": BING_MKT}
    response = requests.get(
        BING_SEARCH_V7_ENDPOINT,
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params=params,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts

def search_with_google(query: str, subscription_key: str, cx: str):
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": REFERENCE_COUNT,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["items"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts

def search_with_serper(query: str, subscription_key: str):
    payload = json.dumps({
        "q": query,
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    })
    headers = {"X-API-KEY": subscription_key, "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SERPER_SEARCH_ENDPOINT}"
    )
    response = requests.post(
        SERPER_SEARCH_ENDPOINT,
        headers=headers,
        data=payload,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = []
        if json_content.get("knowledgeGraph"):
            url = json_content["knowledgeGraph"].get("descriptionUrl") or json_content["knowledgeGraph"].get("website")
            snippet = json_content["knowledgeGraph"].get("description")
            if url and snippet:
                contexts.append({
                    "name": json_content["knowledgeGraph"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        if json_content.get("answerBox"):
            url = json_content["answerBox"].get("url")
            snippet = json_content["answerBox"].get("snippet") or json_content["answerBox"].get("answer")
            if url and snippet:
                contexts.append({
                    "name": json_content["answerBox"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet","")}
            for c in json_content["organic"]
        ]
        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []

def search_with_searchapi(query: str, subscription_key: str):
    payload = {
        "q": query,
        "engine": "google",
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    }
    headers = {"Authorization": f"Bearer {subscription_key}", "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SEARCHAPI_SEARCH_ENDPOINT}"
    )
    response = requests.get(
        SEARCHAPI_SEARCH_ENDPOINT,
        headers=headers,
        params=payload,
        timeout=30,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = []

        if json_content.get("answer_box"):
            if json_content["answer_box"].get("organic_result"):
                title = json_content["answer_box"].get("organic_result").get("title", "")
                url = json_content["answer_box"].get("organic_result").get("link", "")
            if json_content["answer_box"].get("type") == "population_graph":
                title = json_content["answer_box"].get("place", "")
                url = json_content["answer_box"].get("explore_more_link", "")

            title = json_content["answer_box"].get("title", "")
            url = json_content["answer_box"].get("link")
            snippet =  json_content["answer_box"].get("answer") or json_content["answer_box"].get("snippet")

            if url and snippet:
                contexts.append({
                    "name": title,
                    "url": url,
                    "snippet": snippet
                })

        if json_content.get("knowledge_graph"):
            if json_content["knowledge_graph"].get("source"):
                url = json_content["knowledge_graph"].get("source").get("link", "")

            url = json_content["knowledge_graph"].get("website", "")
            snippet = json_content["knowledge_graph"].get("description")

            if url and snippet:
                contexts.append({
                    "name": json_content["knowledge_graph"].get("title", ""),
                    "url": url,
                    "snippet": snippet
                })

        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet", "")}
            for c in json_content["organic_results"]
        ]
        
        if json_content.get("related_questions"):
            for question in json_content["related_questions"]:
                if question.get("source"):
                    url = question.get("source").get("link", "")
                else:
                    url = ""  
                    
                snippet = question.get("answer", "")

                if url and snippet:
                    contexts.append({
                        "name": question.get("question", ""),
                        "url": url,
                        "snippet": snippet
                    })

        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []

class RAG:
    def __init__(self):
        self.backend = os.getenv("BACKEND", "BING").upper()
        self.search_api_key = os.getenv(f"{self.backend}_SEARCH_API_KEY")
        self.cx = os.getenv("GOOGLE_SEARCH_CX", "")
        self.model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.should_do_related_questions = True

        # Setup database
        self.conn = sqlite3.connect('search_results.db')
        self.create_tables()

        # Setup search function based on backend
        self.search_function = self.setup_search_function()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS search_results (
                    search_uuid TEXT PRIMARY KEY,
                    result TEXT
                )
            ''')

    def setup_search_function(self):
        if self.backend == "BING":
            return lambda query: search_with_bing(query, self.search_api_key)
        elif self.backend == "GOOGLE":
            return lambda query: search_with_google(query, self.search_api_key, self.cx)
        elif self.backend == "SERPER":
            return lambda query: search_with_serper(query, self.search_api_key)
        elif self.backend == "SEARCHAPI":
            return lambda query: search_with_searchapi(query, self.search_api_key)
        else:
            raise RuntimeError("Backend must be BING, GOOGLE, SERPER or SEARCHAPI.")

    def get_related_questions(self, query, contexts):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _more_questions_prompt.format(context="\n\n".join([c["snippet"] for c in contexts]))},
                    {"role": "user", "content": query},
                ],
                max_tokens=512,
            )
            related = response.choices[0].message['content']
            return related.strip().split('\n')[:5]
        except Exception as e:
            logger.error(f"Error generating related questions: {e}\n{traceback.format_exc()}")
            return []

    def _raw_stream_response(self, contexts, llm_response, related_questions_future) -> Generator[str, None, None]:
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        if not contexts:
            yield "(The search engine returned nothing for this query. Please take the answer with a grain of salt.)\n\n"
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.get('content', '')
        if related_questions_future is not None:
            related_questions = related_questions_future.result()
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield json.dumps(related_questions)

    def stream_and_upload_to_db(self, contexts, llm_response, related_questions_future, search_uuid) -> Generator[str, None, None]:
        all_yielded_results = []
        for result in self._raw_stream_response(contexts, llm_response, related_questions_future):
            all_yielded_results.append(result)
            yield result
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO search_results (search_uuid, result) VALUES (?, ?)', (search_uuid, ''.join(all_yielded_results)))

    @app.post("/query")
    async def query_function(self, query: str, search_uuid: str, generate_related_questions: Optional[bool] = True) -> StreamingResponse:
        if not search_uuid:
            raise HTTPException(status_code=400, detail="search_uuid must be provided.")
        cur = self.conn.cursor()
        cur.execute('SELECT result FROM search_results WHERE search_uuid = ?', (search_uuid,))
        row = cur.fetchone()
        if row:
            return StreamingResponse(iter(row[0].splitlines()), media_type="text/html")

        query = query or _default_query
        query = re.sub(r"\[/?INST\]", "", query)
        contexts = self.search_function(query)

        system_prompt = _rag_query_text.format(context="\n\n".join([f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]))
        try:
            llm_response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                stop=stop_words,
                stream=True,
                temperature=0.9,
            )
            if self.should_do_related_questions and generate_related_questions:
                related_questions_future = self.executor.submit(self.get_related_questions, query, contexts)
            else:
                related_questions_future = None
        except Exception as e:
            logger.error(f"Error during LLM response generation: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        return StreamingResponse(
            self.stream_and_upload_to_db(contexts, llm_response, related_questions_future, search_uuid),
            media_type="text/html"
        )

    @app.get("/")
    def index():
        return RedirectResponse(url="/ui/index.html")

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

if __name__ == "__main__":
    import uvicorn
    rag = RAG()
    uvicorn.run(app, host="0.0.0.0", port=8000)
import concurrent.futures
import json
import os
import re
import sqlite3
import threading
import traceback
from typing import List, Generator, Optional

import httpx
import openai
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

app = FastAPI()

# Constant values for the RAG model.
BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_MKT = "en-US"
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
SEARCHAPI_SEARCH_ENDPOINT = "https://www.searchapi.io/api/v1/search"

REFERENCE_COUNT = 8
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5

_default_query = "Who said 'live long and prosper'?"

_rag_query_text = """
You are a large language AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

stop_words = [
    "",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

_more_questions_prompt = """
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:
"""

def search_with_bing(query: str, subscription_key: str):
    params = {"q": query, "mkt": BING_MKT}
    response = requests.get(
        BING_SEARCH_V7_ENDPOINT,
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params=params,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts

def search_with_google(query: str, subscription_key: str, cx: str):
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": REFERENCE_COUNT,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["items"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts

def search_with_serper(query: str, subscription_key: str):
    payload = json.dumps({
        "q": query,
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    })
    headers = {"X-API-KEY": subscription_key, "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SERPER_SEARCH_ENDPOINT}"
    )
    response = requests.post(
        SERPER_SEARCH_ENDPOINT,
        headers=headers,
        data=payload,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = []
        if json_content.get("knowledgeGraph"):
            url = json_content["knowledgeGraph"].get("descriptionUrl") or json_content["knowledgeGraph"].get("website")
            snippet = json_content["knowledgeGraph"].get("description")
            if url and snippet:
                contexts.append({
                    "name": json_content["knowledgeGraph"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        if json_content.get("answerBox"):
            url = json_content["answerBox"].get("url")
            snippet = json_content["answerBox"].get("snippet") or json_content["answerBox"].get("answer")
            if url and snippet:
                contexts.append({
                    "name": json_content["answerBox"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet","")}
            for c in json_content["organic"]
        ]
        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []

def search_with_searchapi(query: str, subscription_key: str):
    payload = {
        "q": query,
        "engine": "google",
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    }
    headers = {"Authorization": f"Bearer {subscription_key}", "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SEARCHAPI_SEARCH_ENDPOINT}"
    )
    response = requests.get(
        SEARCHAPI_SEARCH_ENDPOINT,
        headers=headers,
        params=payload,
        timeout=30,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = []

        if json_content.get("answer_box"):
            if json_content["answer_box"].get("organic_result"):
                title = json_content["answer_box"].get("organic_result").get("title", "")
                url = json_content["answer_box"].get("organic_result").get("link", "")
            if json_content["answer_box"].get("type") == "population_graph":
                title = json_content["answer_box"].get("place", "")
                url = json_content["answer_box"].get("explore_more_link", "")

            title = json_content["answer_box"].get("title", "")
            url = json_content["answer_box"].get("link")
            snippet =  json_content["answer_box"].get("answer") or json_content["answer_box"].get("snippet")

            if url and snippet:
                contexts.append({
                    "name": title,
                    "url": url,
                    "snippet": snippet
                })

        if json_content.get("knowledge_graph"):
            if json_content["knowledge_graph"].get("source"):
                url = json_content["knowledge_graph"].get("source").get("link", "")

            url = json_content["knowledge_graph"].get("website", "")
            snippet = json_content["knowledge_graph"].get("description")

            if url and snippet:
                contexts.append({
                    "name": json_content["knowledge_graph"].get("title", ""),
                    "url": url,
                    "snippet": snippet
                })

        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet", "")}
            for c in json_content["organic_results"]
        ]
        
        if json_content.get("related_questions"):
            for question in json_content["related_questions"]:
                if question.get("source"):
                    url = question.get("source").get("link", "")
                else:
                    url = ""  
                    
                snippet = question.get("answer", "")

                if url and snippet:
                    contexts.append({
                        "name": question.get("question", ""),
                        "url": url,
                        "snippet": snippet
                    })

        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []

class RAG:
    def __init__(self):
        self.backend = os.getenv("BACKEND", "BING").upper()
        self.search_api_key = os.getenv(f"{self.backend}_SEARCH_API_KEY")
        self.cx = os.getenv("GOOGLE_SEARCH_CX", "")
        self.model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.should_do_related_questions = True

        # Setup database
        self.conn = sqlite3.connect('search_results.db')
        self.create_tables()

        # Setup search function based on backend
        self.search_function = self.setup_search_function()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS search_results (
                    search_uuid TEXT PRIMARY KEY,
                    result TEXT
                )
            ''')

    def setup_search_function(self):
        if self.backend == "BING":
            return lambda query: search_with_bing(query, self.search_api_key)
        elif self.backend == "GOOGLE":
            return lambda query: search_with_google(query, self.search_api_key, self.cx)
        elif self.backend == "SERPER":
            return lambda query: search_with_serper(query, self.search_api_key)
        elif self.backend == "SEARCHAPI":
            return lambda query: search_with_searchapi(query, self.search_api_key)
        else:
            raise RuntimeError("Backend must be BING, GOOGLE, SERPER or SEARCHAPI.")

    def get_related_questions(self, query, contexts):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _more_questions_prompt.format(context="\n\n".join([c["snippet"] for c in contexts]))},
                    {"role": "user", "content": query},
                ],
                max_tokens=512,
            )
            related = response.choices[0].message['content']
            return related.strip().split('\n')[:5]
        except Exception as e:
            logger.error(f"Error generating related questions: {e}\n{traceback.format_exc()}")
            return []

    def _raw_stream_response(self, contexts, llm_response, related_questions_future) -> Generator[str, None, None]:
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        if not contexts:
            yield "(The search engine returned nothing for this query. Please take the answer with a grain of salt.)\n\n"
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.get('content', '')
        if related_questions_future is not None:
            related_questions = related_questions_future.result()
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield json.dumps(related_questions)

    def stream_and_upload_to_db(self, contexts, llm_response, related_questions_future, search_uuid) -> Generator[str, None, None]:
        all_yielded_results = []
        for result in self._raw_stream_response(contexts, llm_response, related_questions_future):
            all_yielded_results.append(result)
            yield result
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO search_results (search_uuid, result) VALUES (?, ?)', (search_uuid, ''.join(all_yielded_results)))

    @app.post("/query")
    async def query_function(self, query: str, search_uuid: str, generate_related_questions: Optional[bool] = True) -> StreamingResponse:
        if not search_uuid:
            raise HTTPException(status_code=400, detail="search_uuid must be provided.")
        cur = self.conn.cursor()
        cur.execute('SELECT result FROM search_results WHERE search_uuid = ?', (search_uuid,))
        row = cur.fetchone()
        if row:
            return StreamingResponse(iter(row[0].splitlines()), media_type="text/html")

        query = query or _default_query
        query = re.sub(r"\[/?INST\]", "", query)
        contexts = self.search_function(query)

        system_prompt = _rag_query_text.format(context="\n\n".join([f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]))
        try:
            llm_response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                stop=stop_words,
                stream=True,
                temperature=0.9,
            )
            if self.should_do_related_questions and generate_related_questions:
                related_questions_future = self.executor.submit(self.get_related_questions, query, contexts)
            else:
                related_questions_future = None
        except Exception as e:
            logger.error(f"Error during LLM response generation: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        return StreamingResponse(
            self.stream_and_upload_to_db(contexts, llm_response, related_questions_future, search_uuid),
            media_type="text/html"
        )

    @app.get("/")
    def index():
        return RedirectResponse(url="/ui/index.html")

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

if __name__ == "__main__":
    import uvicorn
    rag = RAG()
    uvicorn.run(app, host="0.0.0.0", port=8000)
