"""
Generative debunking in the fact, myth, fallacy, fact ("truth sandwich") structure.

Adapted from the debunking system released by Zanartu et al. (2024),
"Generative Debunking of Climate Misinformation":
https://huggingface.co/spaces/fzanartu/flicc-agent

The structure, prompts and layer logic are theirs. Changes made for this project:

  * the generator is a newer and more token efficient Mistral model served locally with vLLM
  * CARDS and FLICC labels are passed in, because the corpus has already been
    classified upstream by the UK fine tuned model, instead of being predicted here
  * CARDS 2.0 codes are mapped back to the original CARDS labels used by the
    Climate FEVER evidence base
  * evidence retrieval is fixed: the claim label was compared as a number against
    text read from the CSV, so verified claims were not being matched
  * sentence similarity is computed locally instead of through a hosted endpoint
  * the search agent has iteration and time limits, and returns None instead of
    raising when it cannot produce a fact
"""

import csv
import re
import time
from collections import namedtuple
from pathlib import Path

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain_community.llms import VLLMOpenAI
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util

from definitions import DEFINITIONS
from examples import DEBUNKINGS, FALLACY_CLAIMS
from templates import (
    CONCLUDING,
    CONCLUDING_INCONTEXT,
    INCONTEXT,
    REACT,
    SUMMARIZATION,
)

# vLLM endpoint on the GPU machine. See docs/gpu_runs.md.
VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "mistral-small"

# The evidence base is labelled with the original CARDS taxonomy at level 2.
# CARDS 2.0 renumbered some branches, so those codes are mapped back.
CARDS2_TO_CARDS1 = {"7_3": "4_5", "6_2": "5_3", "7_2": "4_5", "7_1": "4_5", "6_1": "5_2"}

# Pulls each evidence sentence out of the stringified evidence list in the Climate FEVER file.
EVIDENCE = re.compile(r"'evidence':\s*'((?:[^'\\]|\\.)*)'")


class HamburgerStyle:
    def __init__(self):
        # hamburger-style structure:
        self.heading = namedtuple("Heading", ["name", "content"])
        self.hamburger = [
            self.heading(name="Myth", content=None),
            self.heading(name="##FACT", content=None),
            self.heading(name="##MYTH", content=None),
            self.heading(name="##FALLACY", content=None),
            self.heading(name="##FACT", content=None),
        ]
        # Completion and chat interfaces to the same locally served model.
        # Temperature 0 so that the output is deterministic.
        self.llm = VLLMOpenAI(openai_api_base=VLLM_URL, openai_api_key="not-needed",
                              model_name=MODEL_NAME, temperature=0)
        self.chat_model = ChatOpenAI(base_url=VLLM_URL, api_key="not-needed",
                                     model=MODEL_NAME, temperature=0)
        self.similarity = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # Climate FEVER claims with CARDS labels, extended by extend_climate_fever.py.
        self.filename = Path(__file__).resolve().parent / "climate_fever_cards.csv"

    def generate_st_layer(self, misinformation):
        ## FACT: ReAct
        prompt = REACT

        # define the agent
        chat_model_with_stop = self.chat_model.bind(stop=["\nObservation"])
        search = DuckDuckGoSearchRun()

        tools = [
            Tool(
                name="google_search",
                description="Search Google for recent results.",
                func=lambda q: str(search.invoke({"query": q})),
            )
        ]

        agent = create_react_agent(chat_model_with_stop, tools, prompt)

        # Hard limits stop the agent looping on statements for which the search
        # returns nothing useful, which otherwise wastes GPU time.
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=False, handle_parsing_errors=True,
            max_iterations=20, max_execution_time=200
        )

        return self.retry_on_exceptions(
            lambda: agent_executor.invoke({"input": misinformation}).get("output")
        )

    def generate_nd_layer(self, misinformation):
        ## MYTH: Summ
        prompt = SUMMARIZATION
        chain = prompt | self.llm
        return chain.invoke({"text": misinformation})

    def generate_rd_layer(self, misinformation, detected_fallacy):
        ## FALLACY: Fallacy
        # 1 the fallacy has already been detected upstream, so only its definition is looked up
        fallacy_definition = DEFINITIONS.get(detected_fallacy)

        # 2 get all examples with the same label
        claims = FALLACY_CLAIMS.get(detected_fallacy, None)

        # 3 get cosine similarity for all claims and myth
        example_myths = self.cosine_similarity(misinformation, claims)

        # 4 get most similar claim and FACT
        max_similarity = example_myths.index(max(example_myths))
        example_myth = claims[max_similarity]
        example_response = DEBUNKINGS.get(claims[max_similarity])
        fact = re.findall(r"## FALLACY:.*?(?=##)", example_response, re.DOTALL)[
            0
        ]  # get only the fallacy layer from the example.
        fact = fact.replace("## FALLACY:", "")

        prompt = INCONTEXT
        chain = prompt | self.chat_model
        content = chain.invoke(
            {
                "misinformation": misinformation,
                "detected_fallacy": detected_fallacy,
                "fallacy_definition": fallacy_definition,
                "example_response": fact,
                "example_myth": example_myth,
                "factual_information": self.hamburger[1].content,
            }
        ).content

        content = re.sub(r"Response:", "", content)

        return content

    def generate_th_layer(self, misinformation, cards_label):
        ## FACT: Concluding
        # 1 get all claims with same label from FEVER dataset
        claims = self.get_fever_claims(cards_label)
        prompt_completition = {"fact": self.hamburger[1].content}
        if claims:
            # 2 use the evidence attached to the most similar verified claim
            prompt = CONCLUDING_INCONTEXT
            example_myths = self.cosine_similarity(misinformation, claims)
            max_similarity = example_myths.index(max(example_myths))
            example_myth = claims[max_similarity]
            complementary_details = self.get_fever_evidence(example_myth)
            prompt_completition.update({"complementary_details": complementary_details})
        else:
            prompt = CONCLUDING

        chain = prompt | self.llm

        return chain.invoke(prompt_completition)

    def rebuttal_generator(self, misinformation, detected_fallacy, cards_label):
        """Build the full debunking for one statement from its precomputed FLICC and CARDS labels."""
        fact = self.generate_st_layer(misinformation)
        # Without an opening fact the remaining layers have nothing to build on.
        if fact is None:
            return None

        # generate rebuttal
        self.hamburger[0] = self.hamburger[0]._replace(content=misinformation)
        ## FACT
        self.hamburger[1] = self.hamburger[1]._replace(content=fact.strip())
        ## MYTH
        self.hamburger[2] = self.hamburger[2]._replace(
            content=self.generate_nd_layer(misinformation).strip()
        )
        ## FALLACY
        self.hamburger[3] = self.hamburger[3]._replace(
            content=self.generate_rd_layer(misinformation, detected_fallacy).strip()
        )
        ## FACT
        self.hamburger[4] = self.hamburger[4]._replace(
            content=self.generate_th_layer(misinformation, cards_label).strip()
        )

        # compose and format the string
        rebuttal = f"""{self.hamburger[1].name}: {self.hamburger[1].content}\n{self.hamburger[2].name}: {self.hamburger[2].content}\n{self.hamburger[3].name}: {self.hamburger[3].content}\n{self.hamburger[4].name}: {self.hamburger[4].content}"""

        return rebuttal

    def cosine_similarity(self, source_sentence, sentences):
        """Cosine similarity between one sentence and each of a list, embedded locally."""
        embeddings = self.similarity.encode([source_sentence] + list(sentences))
        return util.cos_sim(embeddings[0], embeddings[1:])[0].tolist()

    def retry_on_exceptions(self, function, *args):
        """Call a function up to five times with increasing waits. Return None if every attempt fails."""
        attempt = 0
        while attempt < 5:
            try:
                out = function(*args)
                # The agent reports hitting its limits as a string, not an exception.
                if not (isinstance(out, str) and "Agent stopped" in out):
                    return out
            except Exception:
                pass
            print(f"retrying {attempt + 1} out of 5")
            time.sleep(5 * (attempt + 1))
            attempt += 1

        return None

    def get_fever_claims(self, label):
        """Verified Climate FEVER claims that share the level 2 CARDS label of the statement."""
        claims = []
        # Truncate the code to level 2 and map CARDS 2.0 numbering back to the original.
        bits = label.split("_")
        key = bits[0] + "_" + (bits[1] if len(bits) > 1 else "0")
        key = CARDS2_TO_CARDS1.get(key, key)
        with open(self.filename, "r", encoding="utf-8") as csvfile:
            for row in csv.DictReader(csvfile):
                # Values are read as strings, so the label is compared with "1".
                # CARDS_label holds the code followed by its description, hence the prefix match.
                if row["claim_label"] == "1" and row["CARDS_label"].startswith(key + " "):
                    claims.append(row["claim"])
        return claims

    def get_fever_evidence(self, claim):
        """Evidence sentences attached to a verified claim, as a bulleted string."""
        evidences = []
        with open(self.filename, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["claim_label"] == "1" and row["claim"] == claim:
                    evidences += EVIDENCE.findall(row["evidences"])
        return "\n".join("* " + evidence for evidence in evidences)
