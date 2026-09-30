"""Recipes 2/3 data: a STARTER tool-calling dataset with 4 tools (swap in your own tools later).

    python make_tools_data.py
    -> data/tools_main.jsonl    tool calls + the final answer after the tool result
       data/tools_nocall.jsonl  no-tool answers and missing-argument questions (tools list still present)
       data/tools_test.jsonl    HELD-OUT prompts (different phrasings) used by evaluate.py

Why the tools list is in every row, in random order, with distractor tools: the model has to CHOOSE the right
tool from the list each time, not memorize a position.
"""
import json
import random

random.seed(1)

TOOLS = {
    "get_weather": {"description": "Get current weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    "calculate": {"description": "Evaluate a math expression",
                  "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    "set_timer": {"description": "Set a countdown timer",
                  "parameters": {"type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}},
    "web_search": {"description": "Search the web",
                   "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
}


def tools_for(needed=None):
    """Always include the needed tool plus 1-3 distractors, in random order."""
    names = list(TOOLS)
    pick = set(random.sample(names, random.randint(2, 4)))
    if needed:
        pick.add(needed)
    pick = list(pick)
    random.shuffle(pick)
    return [{"type": "function", "function": {"name": n, **TOOLS[n]}} for n in pick]


def call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


CITIES = ["Paris", "Tokyo", "Mumbai", "London", "New York", "Delhi", "Berlin", "Sydney", "Cairo", "Toronto",
          "Dubai", "Madrid", "Rome", "Seoul", "Lagos", "Lima", "Oslo", "Pune", "Goa", "Chennai"]
CONDS = ["sunny", "cloudy", "rainy", "windy", "clear", "foggy"]
TOPICS = ["python tutorials", "best pizza near me", "how volcanoes work", "history of the internet",
          "healthy breakfast ideas", "electric car reviews", "learn guitar online", "space telescope news",
          "cheap flights to goa", "how to fine-tune a model", "latest cricket score", "yoga for beginners"]
SNIPPETS = ["I found several helpful pages about it.", "The top results cover the basics and recent news.",
            "Here are the most relevant results I found."]

W_TRAIN = ["What's the weather in {c}?", "Is it raining in {c}?", "How's the weather in {c} today?",
           "Tell me the temperature in {c}.", "weather {c}", "Will I need an umbrella in {c}?"]
W_TEST = ["Give me the forecast for {c}.", "What's it like outside in {c} right now?"]
C_TRAIN = [("What is {a} times {b}?", "{a}*{b}", "times"), ("Calculate {a} plus {b}.", "{a}+{b}", "plus"),
           ("{a} minus {b} equals?", "{a}-{b}", "minus"), ("What's {a} multiplied by {b}?", "{a}*{b}", "times")]
C_TEST = [("Can you work out {a} * {b}?", "{a}*{b}", "times"), ("Add {a} and {b}.", "{a}+{b}", "plus")]
T_TRAIN = ["Set a timer for {n} minutes.", "Remind me in {n} minutes.", "Start a {n} minute timer.", "Timer {n} min please"]
T_TEST = ["Wake me up in {n} minutes.", "Count down {n} minutes for me."]
S_TRAIN = ["Search the web for {q}.", "Look up {q}.", "Find information about {q}.", "Google {q} for me."]
S_TEST = ["What do people say online about {q}?", "Can you research {q}?"]

NOCALL = {
    "Hi": "Hello! How can I help you today?", "Hello!": "Hi there! What can I do for you?",
    "Hey, how are you?": "I'm doing well, thanks! How can I help?", "Thanks!": "You're welcome!",
    "Good morning": "Good morning! What can I help you with?",
    "What is the capital of France?": "The capital of France is Paris.",
    "Who wrote Romeo and Juliet?": "Romeo and Juliet was written by William Shakespeare.",
    "What color is the sky on a clear day?": "On a clear day the sky looks blue.",
    "How many days are in a week?": "There are seven days in a week.",
    "Tell me a fun fact about cats.": "Cats spend around two thirds of their lives sleeping.",
    "What can you help me with?": "I can check the weather, do math, set timers, and search the web.",
    "Bye": "Goodbye! Have a great day.", "Tell me a short joke.": "Why did the computer get cold? It left its Windows open.",
    "Who are you?": "I'm an assistant that can answer questions and use a few tools.",
}
MISSING = [("What's the weather?", "Which city would you like the weather for?"),
           ("Is it going to rain?", "Which city should I check the weather for?"),
           ("Set a timer.", "For how many minutes should I set the timer?"),
           ("Start a timer please", "How many minutes should the timer run?"),
           ("Search the web.", "What would you like me to search for?"),
           ("Look something up for me.", "What should I look up?"),
           ("Can you calculate something?", "Sure, what would you like me to calculate?"),
           ("Do some math for me.", "Of course, what's the calculation?")]


def weather(tpl):
    c, t, cond = random.choice(CITIES), random.randint(2, 38), random.choice(CONDS)
    return {"tools": tools_for("get_weather"), "messages": [
        {"role": "user", "content": tpl.format(c=c)}, call("get_weather", city=c),
        {"role": "tool", "name": "get_weather", "content": f"{t}C, {cond}"},
        {"role": "assistant", "content": f"It's {t}C and {cond} in {c}."}],
        "expect": {"name": "get_weather", "args": {"city": c}}}


def calc(item):
    tpl, expr, word = item
    a, b = random.randint(2, 99), random.randint(2, 99)
    e = expr.format(a=a, b=b)
    res = eval(e)
    return {"tools": tools_for("calculate"), "messages": [
        {"role": "user", "content": tpl.format(a=a, b=b)}, call("calculate", expression=e),
        {"role": "tool", "name": "calculate", "content": str(res)},
        {"role": "assistant", "content": f"{a} {word} {b} is {res}."}],
        "expect": {"name": "calculate", "args": {"expression": e}}}


def timer(tpl):
    n = random.choice([1, 2, 3, 5, 10, 15, 20, 25, 30, 45, 60])
    return {"tools": tools_for("set_timer"), "messages": [
        {"role": "user", "content": tpl.format(n=n)}, call("set_timer", minutes=n),
        {"role": "tool", "name": "set_timer", "content": "ok"},
        {"role": "assistant", "content": f"Timer set for {n} minutes."}],
        "expect": {"name": "set_timer", "args": {"minutes": n}}}


def search(tpl):
    q = random.choice(TOPICS)
    return {"tools": tools_for("web_search"), "messages": [
        {"role": "user", "content": tpl.format(q=q)}, call("web_search", query=q),
        {"role": "tool", "name": "web_search", "content": random.choice(SNIPPETS)},
        {"role": "assistant", "content": f"Here's what I found about {q}: " + random.choice(SNIPPETS)}],
        "expect": {"name": "web_search", "args": {"query": q}}}


def build(ws, cs, ts, ss, n_each):
    out = []
    for _ in range(n_each):
        out += [weather(random.choice(ws)), calc(random.choice(cs)), timer(random.choice(ts)), search(random.choice(ss))]
    return out


def write(path, rows, keep_expect=False):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r if keep_expect else {k: v for k, v in r.items() if k != "expect"}) + "\n")
    print(f"wrote {len(rows):4d} rows -> {path}")


main = build(W_TRAIN, C_TRAIN, T_TRAIN, S_TRAIN, 40)             # 160 tool-call examples
random.shuffle(main)
nocall = [{"tools": tools_for(), "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
          for q, a in NOCALL.items()]
nocall += [{"tools": tools_for(), "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
           for q, a in MISSING for _ in range(2)]
random.shuffle(nocall)

test = build(W_TEST, C_TEST, T_TEST, S_TEST, 8)                   # 32 held-out call prompts
test_rows = [{"tools": r["tools"], "prompt": r["messages"][0]["content"], "expect": r["expect"]} for r in test]
test_rows += [{"tools": tools_for(), "prompt": q, "expect": None}  # held-out phrasing where NO call is right
              for q in ["Hey!", "What is the largest planet?", "Thank you so much", "Tell me about the ocean."]]

write("data/tools_main.jsonl", main)
write("data/tools_nocall.jsonl", nocall)
with open("data/tools_test.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in test_rows)
print(f"wrote {len(test_rows):4d} rows -> data/tools_test.jsonl  (held out)")
