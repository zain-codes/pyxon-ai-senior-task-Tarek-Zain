"""Tests for individual agents (search, URL, synthesizer).

Agents are tested at the *creation* level — we verify that the factory
functions return properly-configured LangGraph agents with the expected
names, tools, and structure.  No real LLM calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestSearchAgent:
    """Verify ``create_search_agent()`` builds a valid agent."""

    @patch("src.agents.search_agent.create_react_agent")
    @patch("src.agents.search_agent.create_search_tool")
    @patch("src.agents.search_agent.ChatGroq")
    def test_creation(
        self,
        mock_groq: MagicMock,
        mock_tool_factory: MagicMock,
        mock_react: MagicMock,
    ) -> None:
        """Search agent is created with a search tool and correct name."""
        mock_tool = MagicMock(name="tavily_web_search")
        mock_tool_factory.return_value = mock_tool
        mock_model = MagicMock()
        mock_groq.return_value = mock_model
        mock_react.return_value = MagicMock()

        from src.agents.search_agent import create_search_agent

        agent = create_search_agent()

        assert agent is not None
        mock_groq.assert_called_once()
        mock_tool_factory.assert_called_once()
        mock_react.assert_called_once()

        # Verify the agent was created with the right tool and name
        call_kwargs = mock_react.call_args
        assert call_kwargs.kwargs.get("name") == "research_agent"
        assert mock_tool in call_kwargs.kwargs.get("tools", [])


class TestURLAgent:
    """Verify ``create_url_agent()`` builds a valid agent."""

    @patch("src.agents.url_agent.create_react_agent")
    @patch("src.agents.url_agent.create_url_fetch_tool")
    @patch("src.agents.url_agent.ChatGroq")
    def test_creation(
        self,
        mock_groq: MagicMock,
        mock_tool_factory: MagicMock,
        mock_react: MagicMock,
    ) -> None:
        """URL agent is created with a fetch tool and correct name."""
        mock_tool = MagicMock(name="fetch_url")
        mock_tool_factory.return_value = mock_tool
        mock_model = MagicMock()
        mock_groq.return_value = mock_model
        mock_react.return_value = MagicMock()

        from src.agents.url_agent import create_url_agent

        agent = create_url_agent()

        assert agent is not None
        mock_groq.assert_called_once()
        mock_tool_factory.assert_called_once()
        mock_react.assert_called_once()

        # Verify the agent was created with the right tool and name
        call_kwargs = mock_react.call_args
        assert call_kwargs.kwargs.get("name") == "url_agent"
        assert mock_tool in call_kwargs.kwargs.get("tools", [])


class TestSwarmCreation:
    """Verify ``create_swarm()`` wires the supervisor with all agents."""

    @patch("src.swarm.supervisor.create_react_agent")
    @patch("src.swarm.supervisor.create_supervisor")
    @patch("src.swarm.supervisor.create_synthesizer_agent")
    @patch("src.swarm.supervisor.create_url_fetch_tool")
    @patch("src.swarm.supervisor.create_native_search_tool")
    @patch("src.swarm.supervisor.ChatGroq")
    def test_creates_with_three_agents(
        self,
        mock_groq: MagicMock,
        mock_search_tool: MagicMock,
        mock_url_tool: MagicMock,
        mock_synth: MagicMock,
        mock_create_sup: MagicMock,
        mock_react: MagicMock,
    ) -> None:
        """Swarm supervisor is created with research, URL, and synthesizer agents."""
        mock_groq.return_value = MagicMock()
        mock_search_tool.return_value = MagicMock()
        mock_url_tool.return_value = MagicMock()
        mock_synth.return_value = MagicMock(name="synthesizer_agent_mock")
        mock_react.return_value = MagicMock(name="react_agent_mock")

        mock_workflow = MagicMock()
        mock_workflow.compile.return_value = MagicMock()
        mock_create_sup.return_value = mock_workflow

        from src.swarm.supervisor import create_swarm

        app = create_swarm()

        assert app is not None
        mock_create_sup.assert_called_once()

        # Verify 3 agents were passed to create_supervisor
        call_kwargs = mock_create_sup.call_args
        agents_arg = call_kwargs.kwargs.get("agents")
        if agents_arg is None and call_kwargs.args:
            agents_arg = call_kwargs.args[0]
        assert agents_arg is not None
        assert len(agents_arg) == 3
