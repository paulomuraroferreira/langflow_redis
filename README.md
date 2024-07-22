# Redis Chat Memory for a RAG pipeline on Langflow

The repository implements chat memory for a RAG pipeline using Redis, so that past messages are remembered:

![image.png](README_files/redis001.png)

This is done by creating the custom components "Redis message retrieval custom component.json" and "Redis message retrieval custom component.json".

The code for the custom components can be found on the Langflow GUI:

![image.png](README_files/redis002.png)

or below:


Redis Message Retrieval Component


```python
from langflow.base.memory.model import LCChatMemoryComponent
from langflow.inputs import MessageTextInput, SecretStrInput, DictInput, MultilineInput
from langflow.io import MessageTextInput, Output
from langflow.field_typing import BaseChatMessageHistory
import subprocess
import time
from langchain_community.chat_message_histories import RedisChatMessageHistory
import socket

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RedisChatMemory(LCChatMemoryComponent):
    display_name = "Redis Chat Memory"
    description = "Retrieves chat messages from Redis."
    name = "RedisChatMemory"
    icon = "RedisChatMemory"

    inputs = [
        MessageTextInput(
            name="RedisURL",
            display_name="Redis URL (Optional)",
            info="Redis URL",
            required=False,
        ),
        SecretStrInput(
            name="password",
            display_name="Password (Optional)",
            info="User password for the database.",
            required=False,
        ),
        MessageTextInput(
            name="redis_session_id",
            display_name="Redis Session ID (Optional)",
            info="Redis Session ID.",
            advanced=False,
            required=False
        ),
        DictInput(
            name="cluster_kwargs",
            display_name="Cluster arguments",
            info="Optional dictionary of additional keyword arguments for the Redis cluster.",
            advanced=True,
            is_list=True,
        ),
    ]
    
    
    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def start_redis_server(self,port=6379, password=None):
        if self.is_port_in_use(port):
            logging.error(f"Port {port} is already in use. Please choose a different port.")
            return None
    
        cmd = ['redis-server', '--port', str(port)]
        if password:
            cmd.extend(['--requirepass', password])
    
        try:
            redis_process = subprocess.Popen(cmd)
            time.sleep(2)  # Wait for Redis server to start
    
            # Check if the process is still running
            if redis_process.poll() is None:
                logging.info(f"Redis server started successfully on port {port}")
                return redis_process
            else:
                logging.error(f"Failed to start Redis server on port {port}")
                return None
        except Exception as e:
            logging.error(f"Error starting Redis server: {e}")
            return None
    

    def build_message_history(self) -> BaseChatMessageHistory:
        
        session_id = self.redis_session_id or 'default_redis_user_session'

        if self.RedisURL:
            url = self.RedisURL
        else:
            port=6379
            password=self.password or 'admin'
            url = f"redis://default:{password}@localhost:{port}"
            self.start_redis_server(password=password)

        memory = RedisChatMessageHistory(session_id=session_id, url=url)

        return memory
```

Redis Message Storage Component


```python
from langflow.base.memory.model import LCChatMemoryComponent
from langflow.inputs import MessageTextInput, SecretStrInput, DictInput, MultilineInput
from langflow.io import MessageTextInput, Output
from langflow.field_typing import BaseChatMessageHistory
import subprocess
import time
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langflow.schema.message import Message

class RedisChatMemory(Component):
    display_name = "Redis Chat Memory"
    description = "Store chat messages to Redis."
    name = "RedisChatMemory"
    icon = "RedisChatMemory"

    inputs = [
        MessageTextInput(
            name="RedisURL",
            display_name="Redis URL (Optional)",
            info="Redis URL",
            required=False,
        ),
        SecretStrInput(
            name="password",
            display_name="Password (Optional)",
            info="User password for the database.",
            required=False,
        ),
        MessageTextInput(
            name="redis_session_id",
            display_name="Redis Session ID (Optional)",
            info="Redis Session ID.",
            advanced=False,
            required=False
        ),
        DictInput(
            name="cluster_kwargs",
            display_name="Cluster arguments",
            info="Optional dictionary of additional keyword arguments for the Redis cluster.",
            advanced=True,
            is_list=True,
        ),
        MultilineInput(
            name="human_message_input",
            display_name="Human Messages"
            ),
        MessageTextInput(
            name="ai_message_output",
            display_name="AI Message",
            info="RAG Generated answer",
        ),
    ]
    
    outputs = [
    Output(display_name="Message", name="message", method="build_message_history"),
]

    def build_message_history(self) -> Message:
        
 
        session_id = self.redis_session_id or 'default_redis_user_session'

        if self.RedisURL:
            url = self.RedisURL
        else:
            port=6379
            password=self.password or 'admin'
            url = f"redis://default:{password}@localhost:{port}"

        memory = RedisChatMessageHistory(session_id=session_id, url=url)        

        memory.add_user_message(self.human_message_input)
        memory.add_ai_message(self.ai_message_output)
        
        return Message(text = self.ai_message_output)
```

# Tutorial

Clone the repository


```python
git clone https://github.com/paulomuraroferreira/langflow_redis.git
```

Install langflow and redis:


```python
!pip install langflow
!pip install redis
```

On the terminal, execute


```python
langflow run
```

Upload the json 'RAG pipeline with Redis chat memory.json'

![image9.png](README_files/langflow_upload.png)

Copy the pdf documents to the pdf_documents folder,
or change the path in the Document Loader component:

![image2.png](README_files/langflow1.png)

Enter your OpenAI API key on both Embeddings components,

![image3.png](README_files/langflow2.png)

on the OpenAI models component,

![image4.png](README_files/langflow3.png)

Run the chunking pipeline by executing the ChromaDB component:

![Langflow Image2](README_files/langflow9.png)

Enter the playground and ask questions:

![image.png](README_files/redis001.png)

We can see the history on being passed to the LLM by inspecting the prompt component:

![image.png](README_files/redis003.png)
