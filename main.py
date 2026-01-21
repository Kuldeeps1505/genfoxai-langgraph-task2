from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re




load_dotenv()    # Environment setup




class GraphState(TypedDict):     # Graph State Definition
    synopsis: str
    genre: Literal["Drama", "Thriller", "Comedy"]
    story_score: float
    character_score: float
    final_score: float
    reasoning: str



# LLM Initialization

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,  
    api_key=os.getenv("OPENAI_API_KEY"),
)



# Genre Extraction

def extract_genre(text: str) -> Literal["Drama", "Thriller", "Comedy"]:
    normalized = text.strip().lower()

    if normalized == "drama":
        return "Drama"
    if normalized == "thriller":
        return "Thriller"
    if normalized == "comedy":
        return "Comedy"

    # Fallback to safe default to ensure graph termination
    return "Drama"



# Score Extraction

def extract_scores(text: str) -> tuple[float, float]:
    """Extract story and character scores from LLM response."""
    try:
        
        story_match = re.search(r"Story:\s*([\d.]+)", text, re.IGNORECASE)
        character_match = re.search(r"Character:\s*([\d.]+)", text, re.IGNORECASE)
        
        story_score = float(story_match.group(1)) if story_match else 5.0
        character_score = float(character_match.group(1)) if character_match else 5.0
        
        # scores to valid range
        story_score = min(10.0, max(1.0, story_score))
        character_score = min(10.0, max(1.0, character_score))
        
        return story_score, character_score
    except Exception:
        return 5.0, 5.0



# Genre Classifier Node

def genre_classifier(state: GraphState) -> GraphState:
    synopsis = state["synopsis"]

    prompt = f"""
Classify the following movie synopsis into exactly ONE genre.

Allowed genres:
- Drama
- Thriller
- Comedy

Return ONLY the genre name.

Synopsis:
{synopsis}
"""

    try:
        response = llm.invoke(prompt)
        genre = extract_genre(response.content)
    except Exception:
        genre = "Drama"

    return {
        **state,
        "genre": genre,
    }



# Drama Evaluator Node

def drama_evaluator(state: GraphState) -> GraphState:
    synopsis = state["synopsis"]

    prompt = f"""
You are evaluating a DRAMA movie synopsis.

Score the following aspects from 1 to 10:
1. Story quality (emotional depth, meaningful conflict)
2. Character development (complexity, growth)

Return ONLY two numbers in this format:
Story: <score>
Character: <score>

Synopsis:
{synopsis}
"""

    try:
        response = llm.invoke(prompt).content
        story_score, character_score = extract_scores(response)
    except Exception:
        story_score, character_score = 5.0, 5.0

    return {
        **state,
        "story_score": story_score,
        "character_score": character_score,
    }



# Thriller Evaluator node

def thriller_evaluator(state: GraphState) -> GraphState:
    synopsis = state["synopsis"]

    prompt = f"""
You are evaluating a THRILLER movie synopsis.

Score the following aspects from 1 to 10:
1. Story quality (suspense, pacing, stakes)
2. Character development (motivation, tension)

Return ONLY two numbers in this format:
Story: <score>
Character: <score>

Synopsis:
{synopsis}
"""

    try:
        response = llm.invoke(prompt).content
        story_score, character_score = extract_scores(response)
    except Exception:
        story_score, character_score = 5.0, 5.0

    return {
        **state,
        "story_score": story_score,
        "character_score": character_score,
    }



#  Comedy Evaluator node

def comedy_evaluator(state: GraphState) -> GraphState:
    synopsis = state["synopsis"]

    prompt = f"""
You are evaluating a COMEDY movie synopsis.

Score the following aspects from 1 to 10:
1. Story quality (humor setup, payoff)
2. Character appeal (likability, comedic potential)

Return ONLY two numbers in this format:
Story: <score>
Character: <score>

Synopsis:
{synopsis}
"""

    try:
        response = llm.invoke(prompt).content
        story_score, character_score = extract_scores(response)
    except Exception:
        story_score, character_score = 5.0, 5.0

    return {
        **state,
        "story_score": story_score,
        "character_score": character_score,
    }



# Aggregator node

def aggregator(state: GraphState) -> GraphState:
    story = state["story_score"]
    character = state["character_score"]
    genre = state["genre"]

    final_score = round((story + character) / 2, 2)

    # Genre-specific reasoning
    if genre == "Drama":
        reasoning = (
            f"The synopsis presents a compelling dramatic narrative. "
            f"Story quality scored {story}/10 for emotional depth and conflict, "
            f"while character development scored {character}/10 for complexity and growth. "
            f"Final assessment: {final_score}/10."
        )
    elif genre == "Thriller":
        reasoning = (
            f"The synopsis demonstrates thriller elements with strong potential. "
            f"Story quality scored {story}/10 for suspense and pacing, "
            f"while character motivation scored {character}/10 for tension and stakes. "
            f"Final assessment: {final_score}/10."
        )
    else:  # Comedy
        reasoning = (
            f"The synopsis shows comedic promise with solid structure. "
            f"Humor setup and payoff scored {story}/10 for story quality, "
            f"while character appeal scored {character}/10 for likability and comedic potential. "
            f"Final assessment: {final_score}/10."
        )

    return {
        **state,
        "final_score": final_score,
        "reasoning": reasoning,
    }



# Conditional Routing Logic (Graph-Level)

def route_by_genre(state: GraphState) -> str:
    """Route to genre-specific evaluator based on classified genre."""
    if state["genre"] == "Drama":
        return "drama_evaluator"
    if state["genre"] == "Thriller":
        return "thriller_evaluator"
    if state["genre"] == "Comedy":
        return "comedy_evaluator"

    return "drama_evaluator"



# Graph Construction

workflow = StateGraph(GraphState)

workflow.add_node("classifier", genre_classifier)
workflow.add_node("drama_evaluator", drama_evaluator)
workflow.add_node("thriller_evaluator", thriller_evaluator)
workflow.add_node("comedy_evaluator", comedy_evaluator)
workflow.add_node("aggregator", aggregator)

workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    route_by_genre,
    {
        "drama_evaluator": "drama_evaluator",
        "thriller_evaluator": "thriller_evaluator",
        "comedy_evaluator": "comedy_evaluator",
    },
)

workflow.add_edge("drama_evaluator", "aggregator")
workflow.add_edge("thriller_evaluator", "aggregator")
workflow.add_edge("comedy_evaluator", "aggregator")
workflow.add_edge("aggregator", END)

graph = workflow.compile()


if __name__ == "__main__":
    user_synopsis = input("Enter movie synopsis: ").strip()

    if not user_synopsis:
        raise ValueError("Synopsis cannot be empty")

    initial_state: GraphState = {
        "synopsis": user_synopsis, 
        "genre": "Drama",          
        "story_score": 0.0,
        "character_score": 0.0,
        "final_score": 0.0,
        "reasoning": "",
    }

    result = graph.invoke(initial_state)

    # structured output
    
    print("MOVIE SYNOPSIS EVALUATION RESULT")
    
    output = {
        "genre": result["genre"],
        "story_score": result["story_score"],
        "character_score": result["character_score"],
        "final_score": result["final_score"],
        "reasoning": result["reasoning"]
    }
    print(json.dumps(output, indent=2))
    

