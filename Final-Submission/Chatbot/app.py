

from gtts import gTTS
import io
import tempfile
# import soundfile as sf
import os
import plotly
from io import BytesIO
from pathlib import Path
from typing import List

from openai import AsyncAssistantEventHandler, AsyncOpenAI, OpenAI

from literalai.helper import utc_now

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element
from openai.types.beta.threads.runs import RunStep
import dash_bootstrap_components as dbc
import plotly.tools as tls
import plotly.graph_objects as go
import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
# from utils.data_processing import load_data
import openai
# from config import OPENAI_API_KEY
from dash.exceptions import PreventUpdate
from statsmodels.tsa.arima.model import ARIMA
import os

from typing import Optional
from typing import Type
import chainlit.data as cl_data
from chainlit.data.utils import queue_until_user_message
from chainlit.element import Element, ElementDict
from chainlit.socket import persist_user_session
from chainlit.step import StepDict
from literalai.helper import utc_now
import requests
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from typing import Dict, List, Optional
from typing import Type
import chainlit.data as cl_data
from chainlit.data.utils import queue_until_user_message
from chainlit.element import Element, ElementDict
from chainlit.socket import persist_user_session
from chainlit.step import StepDict
from literalai.helper import utc_now


now = utc_now()
existing_data = []
deleted_thread_ids = []  # type: List[str]


# external JavaScript files
external_scripts = [
    'http://localhost:8000/copilot/index.js'
]
# Load and cache data for performance

app = Dash(__name__, external_scripts=external_scripts,external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"
])


async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import os

# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


assistant = sync_openai_client.beta.assistants.retrieve(
    os.environ.get("OPENAI_ASSISTANT_ID")
)


config.ui.name = assistant.name
class CustomDataLayer(cl_data.BaseDataLayer):

    async def upsert_feedback(self, feedback: Feedback) -> str:
    
        new_data = {
            'id': feedback.forId,
            'message': feedback.comment,  # Renamed from 'feedback' to 'comment'
            'status': feedback.value,      # Renamed from 'value' to 'status'
        }
        import os

        payload = {
            "aws_access_key": os.getenv("AWS_ACCESS_KEY"),
            "aws_secret_key": os.getenv("AWS_SECRET_KEY"),
            "region": "us-west-2",
            "table_name": "UniverseFeedback",
            "partition_key": "id",
            "sort_key": "timestamp",
            "data": new_data
        }


        # Send the POST request
        try:
            response = requests.post(
                "http://127.0.0.1:5000/insert-data", json=payload)
            if response.status_code == 200:
                return f"Data successfully sent. Response: {response.json()}"
            else:
                return f"Failed to send data. Status Code: {response.status_code}, Response: {response.text}"
        except requests.RequestException as e:
            return f"An error occurred while sending data: {str(e)}"

        # existing_data.append(new_data)
        # r = requests.post("http://localhost:4050/add", data=new_data)

    async def get_user(self, identifier: str):
        return cl.PersistedUser(id="test", createdAt=now, identifier=identifier)

    async def create_user(self, user: cl.User):
        pass
        # return cl.PersistedUser(id="test", createdAt=now, identifier=user.identifier)

    async def update_thread(
            self,
            thread_id: str,
            name: Optional[str] = None,
            user_id: Optional[str] = None,
            metadata: Optional[Dict] = None,
            tags: Optional[List[str]] = None,
    ):
        thread = next((t for t in existing_data if t["id"] == thread_id), None)
        if thread:
            if name:
                thread["name"] = name
            if metadata:
                thread["metadata"] = metadata
            if tags:
                thread["tags"] = tags
        else:
            existing_data.append(
                {
                    "id": thread_id,
                    "name": name,
                    "metadata": metadata,
                    "tags": tags,
                    "createdAt": utc_now(),
                    "userId": user_id,
                    "userIdentifier": "admin",
                    "steps": [],
                }
            )

    @cl_data.queue_until_user_message()
    async def create_step(self, step_dict: StepDict):
        # print(step_dict)
        pass
        # cl.user_session.set(
        #     "create_step_counter", cl.user_session.get("create_step_counter") + 1
        # )
        #
        # thread = next(
        #     (t for t in existing_data if t["id"] == step_dict.get("threadId")), None
        # )
        # if thread:
        #     thread["steps"].append(step_dict)

    async def get_thread_author(self, thread_id: str):
        return "admin"

    async def list_threads(
            self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        return PaginatedResponse(
            data=[t for t in existing_data if t["id"]
                  not in deleted_thread_ids],
            pageInfo=PageInfo(hasNextPage=False,
                              startCursor=None, endCursor=None),
        )

    async def get_thread(self, thread_id: str):
        thread = next((t for t in existing_data if t["id"] == thread_id), None)
        if not thread:
            return None
        thread["steps"] = sorted(thread["steps"], key=lambda x: x["createdAt"])
        return thread

    async def delete_thread(self, thread_id: str):
        deleted_thread_ids.append(thread_id)

    async def delete_feedback(
            self,
            feedback_id: str,
    ) -> bool:
        return True

    @queue_until_user_message()
    async def create_element(self, element: "Element"):
        pass

    async def get_element(
            self, thread_id: str, element_id: str
    ) -> Optional["ElementDict"]:
        pass

    @queue_until_user_message()
    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass

    @queue_until_user_message()
    async def update_step(self, step_dict: "StepDict"):
        pass

    @queue_until_user_message()
    async def delete_step(self, step_id: str):
        pass

    async def build_debug_url(self) -> str:
        return ""


cl_data._data_layer = CustomDataLayer()

class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

    async def on_run_step_created(self, run_step: RunStep) -> None:
        cl.user_session.set("run_step", run_step)

    async def on_text_created(self, text) -> None:
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    async def on_text_delta(self, delta, snapshot):
        if delta.value:
            await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        await self.current_message.update()
        if text.annotations:
            print(text.annotations)
            for annotation in text.annotations:
                if annotation.type == "file_path":
                    response = await async_openai_client.files.with_raw_response.content(annotation.file_path.file_id)
                    file_name = annotation.text.split("/")[-1]
                    try:
                        fig = plotly.io.from_json(response.content)
                        element = cl.Plotly(name=file_name, figure=fig)
                        await cl.Message(
                            content="",
                            elements=[element]).send()
                    except Exception as e:
                        element = cl.File(content=response.content, name=file_name)
                        await cl.Message(
                            content="",
                            elements=[element]).send()
                    # Hack to fix links
                    if annotation.text in self.current_message.content and element.chainlit_key:
                        self.current_message.content = self.current_message.content.replace(annotation.text, f"/project/file/{element.chainlit_key}?session_id={cl.context.session.id}")
                        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool", parent_id=cl.context.current_run.id)
        self.current_step.show_input = "python"
        self.current_step.start = utc_now()
        await self.current_step.send()

    async def on_tool_call_delta(self, delta, snapshot): 
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool",  parent_id=cl.context.current_run.id)
            self.current_step.start = utc_now()
            if snapshot.type == "code_interpreter":
                 self.current_step.show_input = "python"
            if snapshot.type == "function":
                self.current_step.name = snapshot.function.name
                self.current_step.language = "json"
            await self.current_step.send()
        
        if delta.type == "function":
            pass
        
        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        self.current_step.output += output.logs
                        self.current_step.language = "markdown"
                        self.current_step.end = utc_now()
                        await self.current_step.update()
                    elif output.type == "image":
                        self.current_step.language = "json"
                        self.current_step.output = output.image.model_dump_json()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input, is_input=True)  

    async def on_event(self, event) -> None:
        if event.event == "error":
            return cl.ErrorMessage(content=str(event.data.message)).send()

    async def on_exception(self, exception: Exception) -> None:
        return cl.ErrorMessage(content=str(exception)).send()

    async def on_tool_call_done(self, tool_call):       
        self.current_step.end = utc_now()
        await self.current_step.update()

    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id,
            content=response.content,
            display="inline",
            size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}] if file.mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/markdown", "application/pdf", "text/plain"] else [{"type": "code_interpreter"}],
        }
        for file_id, file in zip(file_ids, files)
    ]


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Run our youth labor data set",
            message="Make a data analysis on the youth_labour_df_updated.csv file I previously uploaded.",
            icon="/public/write.svg",
            ),
        cl.Starter(
            label="Upload your dataset to analysis",
            message="Make a data analysis on the next CSV file I will upload.",
            icon="/public/csv.svg",
            ),
        cl.Starter(
            label="Please upload your resume",
            message="Give me job recommendation and career guidance based on the next resume or cv I will upload",
            icon="/public/resume.svg",
            )
        ]

@cl.on_chat_start
async def start_chat():
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    
    
@cl.on_stop
async def stop_chat():
    current_run_step: RunStep = cl.user_session.get("run_step")
    if current_run_step:
        await async_openai_client.beta.threads.runs.cancel(thread_id=current_run_step.thread_id, run_id=current_run_step.run_id)


# @cl.on_message
# async def main(message: cl.Message):
#     thread_id = cl.user_session.get("thread_id")

#     attachments = await process_files(message.elements)

#     # Add a Message to the Thread
#     oai_message = await async_openai_client.beta.threads.messages.create(
#         thread_id=thread_id,
#         role="user",
#         content=message.content,
#         attachments=attachments,
#     )

#     # Create and Stream a Run
#     async with async_openai_client.beta.threads.runs.stream(
#         thread_id=thread_id,
#         assistant_id=assistant.id,
#         event_handler=EventHandler(assistant_name=assistant.name),
#     ) as stream:
#         await stream.until_done()
        
# @cl.on_message
# async def main(message: cl.Message):
#     thread_id = cl.user_session.get("thread_id")
#     attachments = await process_files(message.elements)

#     # Check if the message is about job recommendations or career guidance
#     if "job recommendation" in message.content.lower() or "career guidance" in message.content.lower():
#         # Process the uploaded resume/CV
#         if attachments:
#             resume_content = "Resume analysis: " + await analyze_resume(attachments[0]["file_id"])
#             message.content += f"\n\n{resume_content}"
#         else:
#             message.content += "\n\nPlease upload your resume or CV for personalized recommendations."

#     # Add a Message to the Thread
#     oai_message = await async_openai_client.beta.threads.messages.create(
#         thread_id=thread_id,
#         role="user",
#         content=message.content,
#         attachments=attachments,
#     )

#     # Create and Stream a Run
#     async with async_openai_client.beta.threads.runs.stream(
#         thread_id=thread_id,
#         assistant_id=assistant.id,
#         event_handler=EventHandler(assistant_name=assistant.name),
#     ) as stream:
#         await stream.until_done()

# async def analyze_resume(file_id):
#     # Implement resume analysis logic here
#     # This could involve using OpenAI's API to extract key information from the resume
#     response = await async_openai_client.files.with_raw_response.content(file_id)
#     resume_text = response.content.decode('utf-8')
    
#     analysis_prompt = f"Analyze the following resume and extract key skills, experience, and education:\n\n{resume_text}"
#     analysis_response = await async_openai_client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": analysis_prompt}]
#     )
    
#     return analysis_response.choices[0].message.content
@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    attachments = await process_files(message.elements)

    # Check if the message is about job recommendations or career guidance
    if "job recommendation" in message.content.lower() or "career guidance" in message.content.lower():
        # Process the uploaded resume/CV
        if attachments:
            resume_content = "Resume analysis: " + await analyze_resume(attachments[0]["file_id"])
            job_recommendations = await get_job_recommendations(resume_content)
            message.content += f"\n\n{resume_content}\n\n{job_recommendations}"
        else:
            message.content += "\n\nPlease upload your resume or CV for personalized recommendations."

    # Add a Message to the Thread
    oai_message = await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments,
    )

    # Create and Stream a Run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name),
    ) as stream:
        await stream.until_done()

async def analyze_resume(file_id):
    response = await async_openai_client.files.with_raw_response.content(file_id)
    resume_text = response.content.decode('utf-8')
    
    analysis_prompt = f"Analyze the following resume and extract key skills, experience, and education:\n\n{resume_text}"
    analysis_response = await async_openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    return analysis_response.choices[0].message.content

async def get_job_recommendations(resume_analysis):
    job_sites = [
        ("Kora Job Portal", "https://jobportal.kora.rw"),
        ("Job in Rwanda", "https://www.jobinrwanda.com"),
        ("Rwanda Job", "https://www.rwandajob.com"),
        ("JobWeb Rwanda", "https://jobwebrwanda.com"),
        ("Bridge2Rwanda", "https://www.bridge2rwanda.org/careers/"),
        ("Harambee Rwanda", "https://harambee.rw")
    ]
    
    recommendation_prompt = f"""Based on the following resume analysis, provide job recommendations and career guidance for a youth in Rwanda. Include relevant links to job websites where they can find opportunities:

Resume Analysis:
{resume_analysis}

Job Websites in Rwanda:
{', '.join([f'{name} ({url})' for name, url in job_sites])}

Provide specific job recommendations, career advice, and mention which websites would be most relevant for the candidate's profile."""

    recommendation_response = await async_openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": recommendation_prompt}]
    )
    
    return recommendation_response.choices[0].message.content




@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[Element]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=transcription, elements=elements)

    await main(message=msg)