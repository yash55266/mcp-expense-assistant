import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from client1 import main

@pytest.mark.asyncio
async def test_main_with_tools():
    with patch('client1.MultiServerMCPClient') as MockClient, \
         patch('client1.ChatOpenAI') as MockChatOpenAI:
        
        # Mock the MultiServerMCPClient
        mock_client_instance = MockClient.return_value
        mock_tool = AsyncMock()
        mock_tool.name = "Summarize"
        mock_client_instance.get_tools.return_value = [mock_tool]
        
        # Mock the ChatOpenAI
        mock_llm_instance = MockChatOpenAI.return_value
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm_instance.ainvoke.side_effect = [
            AsyncMock(tool_calls=[{"name": "Summarize", "id": "1", "args": {}}]),
            AsyncMock(content="Final summarized response")
        ]
        
        # Mock the tool invocation
        mock_tool.ainvoke.return_value = {"summary": "Expenses summary"}
        
        # Run the main function
        await main()
        
        # Assertions
        mock_client_instance.get_tools.assert_called_once()
        mock_llm_instance.bind_tools.assert_called_once_with([mock_tool])
        mock_llm_instance.ainvoke.assert_any_call("summarize all the expenses of all time")
        mock_tool.ainvoke.assert_called_once_with({})
        mock_llm_instance.ainvoke.assert_called_with([
            "summarize all the expenses of all time",
            mock_llm_instance.ainvoke.return_value,
            AsyncMock(content='{"summary": "Expenses summary"}')
        ])

@pytest.mark.asyncio
async def test_main_without_tool_calls():
    with patch('client1.MultiServerMCPClient') as MockClient, \
         patch('client1.ChatOpenAI') as MockChatOpenAI:
        
        # Mock the MultiServerMCPClient
        mock_client_instance = MockClient.return_value
        mock_tool = AsyncMock()
        mock_tool.name = "Summarize"
        mock_client_instance.get_tools.return_value = [mock_tool]
        
        # Mock the ChatOpenAI
        mock_llm_instance = MockChatOpenAI.return_value
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm_instance.ainvoke.return_value = AsyncMock(tool_calls=None, content="LLM reply without tool calls")
        
        # Run the main function
        await main()
        
        # Assertions
        mock_client_instance.get_tools.assert_called_once()
        mock_llm_instance.bind_tools.assert_called_once_with([mock_tool])
        mock_llm_instance.ainvoke.assert_called_once_with("summarize all the expenses of all time")