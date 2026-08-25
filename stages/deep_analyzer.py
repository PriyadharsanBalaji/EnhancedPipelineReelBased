"""
stages/deep_analyzer.py — Stage 1: Deep concept extraction via Ollama LLM.

Takes the extracted chapter text and produces a comprehensive analysis covering:
  - Ancient history & origins of the topic
  - Core concepts & definitions
  - Common misconceptions
  - Fun facts & surprising connections
  - Tricks & shortcuts for exams
  - Worked examples with step-by-step solutions
  - Visual animation ideas for each concept
"""

import json
import re
from pathlib import Path

import ollama

from config import OLLAMA_BASE_URL, STORYBOARD_MODEL, STORYBOARD_TEMPERATURE


DEEP_ANALYSIS_SYSTEM = """You are an expert educational content creator who specializes in making complex subjects incredibly exciting and curious for children and young teenagers.
You have deep knowledge of pedagogy and how to make abstract concepts feel like magic tricks or thrilling discoveries.

Your task: Analyze the provided chapter text and extract the most mind-blowing, curious, and engaging "Aha!" moments.
DO NOT output boring textbook definitions. DO NOT output long historical backgrounds. Instead, focus entirely on the "cool factor", surprising connections, and visual intuition that will make a kid say 'Wow!'

You must output ONLY valid JSON. No markdown fences, no preamble, no extra text."""


DEEP_ANALYSIS_PROMPT = """Analyze this chapter thoroughly and return a highly engaging, curiosity-driven breakdown tailored for a kid.

CHAPTER TEXT:
{chapter_text}

SECTIONS FOUND:
{sections_summary}

DEFINITIONS FOUND:
{definitions_summary}

EXAMPLES FOUND:
{examples_summary}

Return this EXACT JSON structure:
{{
    "chapter_title": "Full chapter title",
    "subject": "Mathematics",
    "grade": "Class 11",
    "topic_summary": "1 sentence: The absolute coolest thing about this chapter.",

    "mind_blowing_hooks": [
        {{
            "hook": "A crazy question or paradox to grab a kid's attention instantly",
            "why_it_works": "Why this will make them curious",
            "visual_idea": "How to show this visually (e.g. a bulging number, a shrinking galaxy)"
        }}
    ],

    "core_concepts_intuition": [
        {{
            "name": "Concept name (e.g. 'Roster Form', 'Union of Sets')",
            "intuition": "Everyday analogy or intuitive explanation that a kid would instantly get",
            "visual_idea": "Specific Manim objects, transformations, and dynamic emphasis (bulging, wiggling, glowing)",
            "why_it_is_cool": "The 'aha!' moment of this concept"
        }}
    ],

    "misconceptions_debunked": [
        {{
            "wrong_belief": "What kids commonly get wrong",
            "visual_fix": "How a dynamic animation could demonstrate the correct idea (e.g., smashing the wrong answer)"
        }}
    ],

    "fun_facts": [
        {{
            "fact": "A genuinely surprising, kid-friendly fact",
            "wow_factor": "Why this is mind-blowing"
        }}
    ],

    "magic_tricks_and_shortcuts": [
        {{
            "name": "Cool name for the trick",
            "description": "How to use this trick to solve problems super fast",
            "example": "Quick example showing the trick in action"
        }}
    ],

    "chapter_flow": [
        "Ordered list of topics as they should be taught for maximum curiosity and engagement"
    ]
}}

CRITICAL INSTRUCTIONS:
- NO BORING HISTORY. Skip all ancient origins. We want modern, punchy, exciting content.
- For mind_blowing_hooks: Include AT LEAST 3 amazing hooks.
- For core_concepts_intuition: Cover EVERY concept but explain it like a magic trick or game mechanic.
- For visual_idea fields: Be SPECIFIC about Manim objects (Circle, VGroup, MathTex, NumberLine). Explicitly mention dynamic emphasis (bulging, scaling, wiggling, glowing) for important elements.
- Make everything highly accessible, intellectually stimulating, and FUN for a child."""


def analyze_chapter(
    chapter_content: dict,
    output_path: str = None,
    model: str = None,
) -> dict:
    """
    Send chapter text to LLM for deep analysis.

    Args:
        chapter_content: Output from pdf_extractor.extract_pdf()
        output_path: Save analysis JSON here
        model: Override storyboard model

    Returns:
        Deep analysis dict
    """
    model = model or STORYBOARD_MODEL

    # Build context for the LLM
    full_text = chapter_content.get("full_text", "")
    # Truncate to fit context window (~30k chars should be safe for 128K context)
    if len(full_text) > 30000:
        full_text = full_text[:30000] + "\n\n[... truncated ...]"

    sections_summary = "\n".join(
        f"  {s['number']} — {s['title']}"
        for s in chapter_content.get("sections", [])
    )

    definitions_summary = "\n".join(
        f"  - {d[:200]}" for d in chapter_content.get("definitions", [])
    )

    examples_summary = "\n".join(
        f"  Example {e['number']}: {e['content'][:150]}..."
        for e in chapter_content.get("examples", [])
    )

    prompt = DEEP_ANALYSIS_PROMPT.format(
        chapter_text=full_text,
        sections_summary=sections_summary or "(none detected — use full text)",
        definitions_summary=definitions_summary or "(none detected — extract from text)",
        examples_summary=examples_summary or "(none detected — extract from text)",
    )

    print(f"[Analyzer] Sending {len(prompt)} chars to {model}...")
    print(f"[Analyzer] This may take 2-5 minutes for a thorough analysis...")

    client = ollama.Client(host=OLLAMA_BASE_URL)

    messages = [
        {"role": "system", "content": DEEP_ANALYSIS_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": STORYBOARD_TEMPERATURE,
                "num_predict": 16384,  # Allow long response
            },
            format="json",
        )

        text = response.get("message", {}).get("content", "").strip()

        try:
            analysis = _robust_json_parse(text)
            _validate_analysis(analysis)
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to generate valid deep analysis after 3 attempts: {e}")
            print(f"[Analyzer] Warning: LLM produced invalid JSON ({e}). Asking it to self-correct... (Attempt {attempt+1}/3)")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Your previous output was invalid JSON or failed validation: {e}. Please fix the formatting/content and return ONLY valid JSON."})

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"[Analyzer] Saved -> {output_path}")

    _print_summary(analysis)
    return analysis


def _validate_analysis(analysis: dict) -> None:
    """Validate the analysis has required fields."""
    required = ("core_concepts_intuition", "mind_blowing_hooks", "chapter_flow")
    for key in required:
        if key not in analysis:
            raise ValueError(f"Analysis missing required field: '{key}'")

    if len(analysis.get("core_concepts_intuition", [])) < 3:
        print(f"[Analyzer] WARNING: Only {len(analysis.get('core_concepts_intuition', []))} core concepts found — expected more")

    if len(analysis.get("mind_blowing_hooks", [])) < 3:
        print(f"[Analyzer] WARNING: Only {len(analysis.get('mind_blowing_hooks', []))} hooks found — expected more")


def _print_summary(analysis: dict) -> None:
    """Print a summary of the analysis."""
    print(f"\n{'='*60}")
    print(f"  Deep Analysis Complete (Curiosity Mode)")
    print(f"{'='*60}")
    print(f"  Chapter:        {analysis.get('chapter_title', 'Unknown')}")
    print(f"  Core Intuitions:{len(analysis.get('core_concepts_intuition', []))}")
    print(f"  Mind-Blowing:   {len(analysis.get('mind_blowing_hooks', []))}")
    print(f"  Misconceptions: {len(analysis.get('misconceptions_debunked', []))}")
    print(f"  Fun facts:      {len(analysis.get('fun_facts', []))}")
    print(f"  Magic Tricks:   {len(analysis.get('magic_tricks_and_shortcuts', []))}")
    print(f"  Chapter flow:   {len(analysis.get('chapter_flow', []))} steps")
    print(f"{'='*60}\n")


def _robust_json_parse(text: str) -> dict:
    """
    Robustly parse JSON generated by LLMs, fixing common syntax issues:
    - Missing commas between fields/objects
    - Unescaped newlines inside strings
    - Truncated JSON near the end
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r'("\s*|\d+|true|false|null|\]|\})\s*\n\s*("|\{|\[)', r'\1,\n\2', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r',\s*([\}\]])', r'\1', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    open_brackets = 0
    open_braces = 0
    in_string = False
    escape = False

    cleaned_chars = []
    for char in fixed:
        cleaned_chars.append(char)
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            elif char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1

    if in_string:
        cleaned_chars.append('"')

    res_str = "".join(cleaned_chars).rstrip()
    if res_str.endswith(','):
        res_str = res_str[:-1]

    res_str += ']' * max(0, open_brackets)
    res_str += '}' * max(0, open_braces)

    try:
        return json.loads(res_str)
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Failed to parse LLM JSON (len {len(text)})", text, 0)
