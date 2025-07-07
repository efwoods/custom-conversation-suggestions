from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Dict
import json


class ConversationSuggestionEngine:
    def __init__(self, model_name: str = "llama2"):
        """Iinitialize the conversation suggestion engine with local Llama model.

        Args:
            model_name (str, optional): This is the model to be used with ollama. Defaults to "llama2".
        """
        self.llm = Ollama(model=model_name)

        # Memory to keep track of conversation context
        self.memory = ConversationBufferWindowMemory(
            k=5, return_messages=True  # Keep last 5 exchanges
        )

        # Prompt template for generating converstaion suggestions
        self.suggestion_prompt = PromptTemplate(
            input_variables=[
                "conversation_history",
                "current_topic",
                "user_preferences",
            ],
            template="""
            Based on the conversation history and current context, generate 3 relevant conversation suggestions.

            Conversation History:
            {conversation_history}

            Current Topic: {current_topic}

            User Preferences: {user_preferences}

            Generate exactly 3 conversation suggestions that would naturally continue this conversation.
            Each suggestion should be engaging, relevant, and different from the others.

            Format your response as a JSON array of strings:
            ["suggestion 1", "suggestion 2", "suggestion 3"]
            """,
        )

        # Create the chain
        self.suggestion_chain = LLMChain(
            llm=self.llm, prompt=self.suggestion_prompt, memory=self.memory
        )

    def generate_suggestions(
        self,
        conversation_history: str,
        current_topic: str = "",
        user_preferences: str = "",
    ) -> List[str]:
        """Generate conversation suggestions based on context.
        Args:
            conversation_history (str): This is the current history of messages.
            current_topic (str, optional): This is the current topic of conversation. Defaults to "".
            user_preferences (str, optional): These are the settings to the specific avatar. Defaults to "".
        Returns:
            List[str]: Returns a list of conversations suggestions.
        """
        try:
            # Generate suggestions using the chain
            response = self.suggestion_chain.run(
                conversation_history=conversation_history,
                current_topic=current_topic,
                user_preferences=user_preferences,
            )
            # Parse the JSON response
            suggestions = json.loads(response.strip())
            # Ensure we have a list of strings
            if isinstance(suggestions, list):
                return suggestions[:3]  # Limit to 3 suggestions
            else:
                return [response]  # Fallback if JSON parsing fails
        except json.JSONDecodeError:
            # Fallback: split by newlines and clean up
            lines = response.strip().split("\n")
            suggestions = []
            for line in lines:
                cleaned = line.strip().strip('"-').strip()
                if (
                    cleaned
                    and not cleaned.startswith("[")
                    and not cleaned.startswith("]")
                ):
                    suggestions.append(cleaned)
            return suggestions[:3]
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return [
                "Let's change the topic.",
                "Expand on that idea.",
                "How did that make you feel?",
            ]

    def add_to_conversation(self, user_input: str, avatar_response: str):
        """Add exchange to conversation memory.
        Args:
            user_input (str): This is the input from the user.
            avatar_response (str): This is the response from the avatar.
        """
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(avatar_response)

    def get_contextual_suggestions(
        self,
        recent_messages: List[Dict[str, str]],
        topic: str = "",
        user_interests: List[str] = None,
    ) -> List[str]:
        """Generate contextural suggestions based on recent conversation.
        Args:
            recent_messages (List[Dict[str, str]]): These are the recent messages.
            topic (str, optional): This is the topic of conversation. Defaults to "".
            user_interests (List[str], optional): These are the user preferences. Defaults to None.
        Returns:
            List[str]: _description_
        """
        # Format conversation history
        history = ""
        for msg in recent_messages[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history += f"{role}: {content}\n"
        # Format user preferences
        preferences = ""
        if user_interests:
            preferences = f"User is interested in: {', '.join(user_interests)}"
        # Current context
        context = f"Current topic: {topic}" if topic else "General conversation"
        return self.generate_suggestions(
            conversation_history=history,
            current_topic=context,
            user_preferences=preferences,
        )
