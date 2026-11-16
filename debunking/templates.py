"""
Prompt templates for each layer of the debunking, from Zanartu et al. (2024).
"""

from langchain.prompts import PromptTemplate

REACT = PromptTemplate.from_template(
    """<s>[INST] You will receive a piece of misinformation related to climate change. Your task is to translate this misinformation into a climate change-related question that challenges the misinformation and prompts for factual investigation.
For example, if the misinformation is: "Climate change isn't real because it's been cold this winter."
The translated question could be: "How does winter weather in one location relate to the broader scientific consensus on climate change?"
Be as specific as possible, ensuring the question directly addresses climate change and encourages factual investigation. You have access to the following tools:

{tools}
[/INST]
</s>
[INST]
Use the following format:

Question: the translated question challenging the misinformation and prompting for factual investigation
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: provide a factual response to the original misinformation, limit your answer two sentences or less than 30 words. Be specific, prefer facts that contain numbers or are backed up by recognised institutions or climate experts to ensure credibility.

Begin!
[/INST]
Question: {input}
Thought:{agent_scratchpad}"""
)

INCONTEXT = PromptTemplate.from_template(
    """<s>[INST] <<SYS>>
You are a senior climate analyst, an expert in identifying and responding to climate change misinformation.
<</SYS>>
What fallacy is contained in the following climate change misinformation?
misinformation: {misinformation} [/INST] 
Your text contains {detected_fallacy} fallacy. {detected_fallacy} fallacy is {fallacy_definition} </s>
<s>[INST] What is the factual evidence surrounding this climate change misinformation?[/INST]
{factual_information}</s>
<s>[INST] Provide a precise and concise response to this climate change misinformation.
In two sentences, explicitly name the fallacy, explain why it's incorrect, and link it to factual evidence showing how it distorts reality.
Consider the following example before providing your answer:
Misinformation: {example_myth}
Response: {example_response}
Misinformation: {misinformation}
Response: 
[/INST]"""
)

SUMMARIZATION = PromptTemplate.from_template(
    """[INST]
You are a paraphrasing system capable of providing rephrased versions of texts in clear and concise language.
Paraphrase the following text in 30 words or fewer. Only refer to the text without adding additional elements or opinions.
[/INST]

text: {text}
Summary: 
"""
)

CONCLUDING = PromptTemplate.from_template(
    """<s>[INST] 1. Reinforce the following fact and provide complementary details, if relevant, to enhance understanding.
2. The output should be simple text summarizing the information in 30 words or fewer. [/INST]
</s>
# Fact:
{fact}
# Summary:
"""
)

CONCLUDING_INCONTEXT = PromptTemplate.from_template(
    """<s>[INST] 1. Reinforce the following fact and provide complementary details, if relevant, to enhance understanding.
2. The output should be simple text summarizing the information in 30 words or fewer. Replace technical and complex words with simpler synonyms and delete unimportant information.[/INST]
Complementary details:
{complementary_details}
</s>

# Fact:
{fact}
# Summary:
"""
)
