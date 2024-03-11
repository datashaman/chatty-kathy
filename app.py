"""
Experiments with griptape and gradio.
"""

import os

import gradio as gr

from dotenv import load_dotenv
from griptape.engines import VectorQueryEngine
from griptape.loaders import WebLoader
from griptape.memory.structure import Run
from griptape.rules import Ruleset, Rule
from griptape.structures import Agent
from griptape.tools import VectorStoreClient
from griptape.drivers import (
    LocalVectorStoreDriver,
    OpenAiEmbeddingDriver,
    OpenAiChatPromptDriver,
)

load_dotenv()

namespace = os.getenv("APP_NAMESPACE", "griptape")

embedding_driver = OpenAiEmbeddingDriver(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

prompt_driver = OpenAiChatPromptDriver(
    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
    temperature=os.getenv("OPENAI_TEMPERATURE"),
)

engine = VectorQueryEngine(
    prompt_driver=prompt_driver,
    vector_store_driver=LocalVectorStoreDriver(embedding_driver=embedding_driver),
)

artifacts = WebLoader().load("https://en.wikipedia.org/wiki/Physics")

engine.vector_store_driver.upsert_text_artifacts({namespace: artifacts})

vector_store_tool = VectorStoreClient(
    description="Contains information about physics. "
    "Use it to answer any physics-related questions.",
    query_engine=engine,
    namespace=namespace,
    off_prompt=False,
)


def chat(_input, history):
    "Respond to input from the user, given history of the conversation."
    agent = Agent(
        rulesets=[
            Ruleset(
                name="Physics Tutor",
                rules=[
                    Rule("Always introduce yourself as a physics tutor"),
                    Rule("Be truthful. Only discuss physics."),
                ],
            )
        ],
        tools=[vector_store_tool],
    )
    _ = [
        agent.conversation_memory.try_add_run(Run(input=entry[0], output=entry[1]))
        for entry in history
    ]
    yield agent.run(_input).output_task.output.to_text()


gr.ChatInterface(chat).launch()
