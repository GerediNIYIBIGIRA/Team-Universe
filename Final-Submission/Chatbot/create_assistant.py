from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


instructions = """You are an assistant running data analysis on CSV files, remember you have developed by the Team of students From Carnegie Mellon University namely Geredi Niyibigira, Odile Nzambazamariya, Ngoga Alexis.
when you are welcomming the user use this welcome to Team Universe employment AI Assistance.

You will use code interpreter to run the analysis.

However, instead of rendering the charts as images, you will generate a plotly figure and turn it into json.
I also be ready and able to provide the insights to the graph you have generated.
also you re a career guidance assistance to Job seeker, if the user asked the question related to career guidance or job opportunities please use file search to give him the job available on the job websites mainly focuse on those which are locally
To guise the user well regarding to job opportunities or career guidance ask the user to upload his/her resume/cv in order to provide him/her the well good answer by using File search, please ask the use to upload the the resume or cv to give job recommendention and career guidance
You will create a file for each json that I can download through annotations.
"""

tools = [
    {"type": "code_interpreter"},
    {"type": "file_search"}
]

file = openai_client.files.create(
  file=open("youth_labour_df_updated.csv", "rb"),
  purpose='assistants'
)


assistant = openai_client.beta.assistants.create(
    model="gpt-4o",
    # gpt-3.5-turbo
    
    name="Team Universe employment AI",
    instructions=instructions,
    temperature=0.1,
    tools=tools,
    tool_resources={
        "code_interpreter": {
        "file_ids": [file.id]
        }
    }
)

print(f"Assistant created with id: {assistant.id}")
# from dotenv import load_dotenv
# load_dotenv()

# import os
# import requests
# from bs4 import BeautifulSoup
# from openai import OpenAI

# openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# # Scrape websites
# def scrape_website(url):
#     response = requests.get(url)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, 'html.parser')
#         return soup.get_text()
#     else:
#         return f"Failed to fetch {url}, Status Code: {response.status_code}"

# scraped_links = [
#     "https://neveragainrwanda.org/youth-unemployment-and-perplexing-access-to-finance-in-rwanda/",
#     "https://statistics.gov.rw/publication/2119",
#     "https://www.statistics.gov.rw/publication/2138"
# ]

# scraped_content = [scrape_website(url) for url in scraped_links]

# # Read the TXT file
# def read_txt_file(file_path):
#     with open(file_path, 'r') as file:
#         return file.read()

# txt_file_path = 'summary_statistics_all.txt'  # Update this path if needed
# txt_file_content = read_txt_file(txt_file_path)

# # Combine data sources into a single string for instructions
# combined_text = "\n\n".join([
#     "### Web Scraped Content:",
#     *scraped_content,
#     "### TXT File Content:",
#     txt_file_content
# ])

# instructions = f"""
# You are an assistant for data analysis and information retrieval.

# You can:
# 1. Run data analysis on CSV files using a code interpreter.
# 2. Retrieve and analyze the following information:
# {combined_text}

# For data analysis, use Plotly to generate JSON outputs. For web scraping and TXT files, summarize the key points or generate JSON files for further processing.
# """

# tools = [
#     {"type": "code_interpreter"},
#     {"type": "file_search"}
# ]

# # Upload the CSV file
# file = openai_client.files.create(
#   file=open("youth_labour_df_updated.csv", "rb"),
#   purpose='assistants'
# )

# # Create assistant
# assistant = openai_client.beta.assistants.create(
#     model="gpt-4o",
#     name="Data Analysis Assistant",
#     instructions=instructions,
#     temperature=0.1,
#     tools=tools,
#     tool_resources={
#         "code_interpreter": {"file_ids": [file.id]},
#     }
# )

# print(f"Assistant created with id: {assistant.id}")
