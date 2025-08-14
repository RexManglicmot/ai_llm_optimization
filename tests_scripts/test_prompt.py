
from app.prompt import build_prompt_numeric, parse_index
# Quick sanity test for prompt building & parsing

# 1) Build a numeric-labeled prompt for the question.
#    This is just for debugging to make sure the function works as intended.
#    Correct answer to this question:
#       "Which vitamin deficiency causes scurvy?"
#       → Vitamin C → index 2 (if 0=Vitamin A, 1=Vitamin B12, 2=Vitamin C, 3=Vitamin D)
p = build_prompt_numeric("Which vitamin deficiency causes scurvy?", ["A", "B", "C", "D"])

# 2) Check that the prompt contains the 'Answer:' cue AND see its length.
#    This check is NOT needed in production — it’s only for debugging.
#    Expected output for first print: True <character length of prompt>
print("Answer:" in p, len(p))

# 3) Test parse_index() with different forms of numeric answers.
#    This ensures the parser can find the correct choice index no matter how it's formatted in the text.
#    Example question's correct answer index: 2
#    Expected output for second print: 2 1 3
print(
    parse_index("2"),                       # Direct numeric answer
    parse_index("the answer is 1"),         # Number in a sentence
    parse_index("I choose 3) because...")   # Number before a parenthesis
)




# Run: Python3 -m tests_scripts.test_prompt