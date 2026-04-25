import unittest
from unittest.mock import patch, MagicMock
import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class TestClient2(unittest.TestCase):

    @patch('client2.ChatOpenAI')
    @patch('client2.MultiServerMCPClient')
    @patch('client2.st')
    def test_initialization(self, mock_st, mock_MultiServerMCPClient, mock_ChatOpenAI):
        # Mock the Streamlit session state
        mock_st.session_state = {}
        
        # Mock the ChatOpenAI and MultiServerMCPClient
        mock_llm = MagicMock()
        mock_ChatOpenAI.return_value = mock_llm
        mock_client = MagicMock()
        mock_MultiServerMCPClient.return_value = mock_client
        mock_client.get_tools.return_value = asyncio.Future()
        mock_client.get_tools.return_value.set_result([])

        # Import the module to trigger initialization
        import client2

        # Check if session state is initialized correctly
        self.assertTrue(mock_st.session_state['initialized'])
        self.assertIsInstance(mock_st.session_state['history'][0], SystemMessage)
        self.assertEqual(mock_st.session_state['history'][0].content, client2.SYSTEM_PROMPT)

    @patch('client2.st')
    def test_user_input(self, mock_st):
        # Mock the Streamlit session state
        mock_st.session_state = {
            'initialized': True,
            'history': [SystemMessage(content="System Prompt")]
        }
        
        # Mock user input
        mock_st.chat_input.return_value = "Test input"

        # Import the module to trigger user input handling
        import client2

        # Check if user input is appended to history
        self.assertIsInstance(mock_st.session_state['history'][1], HumanMessage)
        self.assertEqual(mock_st.session_state['history'][1].content, "Test input")

    @patch('client2.st')
    @patch('client2.asyncio.run')
    def test_tool_call_handling(self, mock_asyncio_run, mock_st):
        # Mock the Streamlit session state
        mock_st.session_state = {
            'initialized': True,
            'history': [SystemMessage(content="System Prompt")],
            'llm_with_tools': MagicMock(),
            'tool_by_name': {'tool_name': MagicMock()}
        }
        
        # Mock the tool call and response
        mock_tool_call = {'name': 'tool_name', 'id': '123', 'args': '{}'}
        mock_asyncio_run.side_effect = [
            AIMessage(content="", tool_calls=[mock_tool_call]),
            {'result': 'tool result'},
            AIMessage(content="Final response")
        ]

        # Import the module to trigger tool call handling
        import client2

        # Check if tool call and final response are handled correctly
        self.assertEqual(len(mock_st.session_state['history']), 4)
        self.assertIsInstance(mock_st.session_state['history'][2], AIMessage)
        self.assertEqual(mock_st.session_state['history'][3].content, "Final response")

if __name__ == '__main__':
    unittest.main()