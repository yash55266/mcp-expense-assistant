import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import json

load_dotenv()

SERVERS={
    "Summarize":{
        "transport":"stdio",
        "command":'uv',
        "args":[
            "run",
            "fastmcp",
            "run",
            r"C:\Users\Yash\Desktop\sample_project_1\expense-tracker\main.py"

        ]
    }
}


async def main():
    client=MultiServerMCPClient(SERVERS)
    tools= await client.get_tools()
    named_tools={}
    for tool in tools:
        named_tools[tool.name]=tool

    llm=ChatOpenAI(model="gpt-5")
    llm_with_tools=llm.bind_tools(tools)

    prompt="summarize all the expenses of all time"
    response = await llm_with_tools.ainvoke(prompt)
    print("reppppp",response.tool_calls)
    if not getattr(response, "tool_calls", None):
        print("LLM reply",response.content)
        return

    tool_messages=[]
    for tc in response.tool_calls:
        selected_tool=tc["name"]
        selected_tool_args=tc.get('args') or {}
        selected_tool_id=tc["id"]
        result= await named_tools[selected_tool].ainvoke(selected_tool_args)

        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id,content=json.dumps(result)))

    

    final_repsonse= await llm_with_tools.ainvoke([prompt,response,*tool_messages])

    print("Final response",final_repsonse.content)

if __name__=='__main__':
    asyncio.run(main())