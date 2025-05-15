"""Claims extraction and ranking module."""

import math
from collections.abc import Iterable
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, chain
from pydantic import BaseModel, Field

class Claim(BaseModel):
    """Representation of an individual claim from a source document(s)."""

    claim: str = Field(description="The claim from the original document(s).")
    source_id: str = Field(description="Document ID containing the claim.")

class Claims(BaseModel):
    """Claims extracted from a set of source document(s)."""

    claims: list[Claim] = Field(description="The extracted claims.")

class ClaimsChainInput(TypedDict):
    question: str
    communities: Iterable[Iterable[Document]]

class ClaimsProcessor:
    """Processes and ranks claims from documents."""

    def __init__(self, model: BaseLanguageModel) -> None:
        """Initialize the claims processor.
        
        Args:
            model: The language model to use for processing
        """
        self.model = model
        self.claims_model = model.with_structured_output(Claims)
        
        # Initialize prompts
        self.claims_prompt = ChatPromptTemplate.from_template("""
Extract claims from the following related documents.

Only return claims appearing within the specified documents.
If no documents are provided, do not make up claims or documents.

Claims (and scores) should be relevant to the question.
Don't include claims from the documents if they are not directly or indirectly
relevant to the question.

If none of the documents make any claims relevant to the question, return an
empty list of claims.

If multiple documents make similar claims, include the original text of each as
separate claims. Score the most useful and authoritative claim higher than
similar, lower-quality claims.

Question: {question}

{formatted_documents}
""")

        self.rank_prompt = ChatPromptTemplate.from_template("""
Rank the relevance of the following claim to the question.
Output "True" if the claim is relevant and "False" if it is not.
Only output True or False.

Question: Where is Seattle?

Claim: Seattle is in Washington State.

Relevant: True

Question: Where is LA?

Claim: New York City is in New York State.

Relevant: False

Question: {question}

Claim: {claim}

Relevant:
""")

    def format_documents_with_ids(self, documents: Iterable[Document]) -> str:
        """Format documents with their IDs for the prompt.
        
        Args:
            documents: Documents to format
            
        Returns:
            str: Formatted document string
        """
        formatted_docs = "\n\n".join(
            f"Document ID: {doc.id}\nContent: {doc.page_content}" for doc in documents
        )
        return formatted_docs

    def compute_rank(self, msg) -> float:
        """Compute rank score from model response.
        
        Args:
            msg: Model response message
            
        Returns:
            float: Rank score between 0 and 1
        """
        logprob = msg.response_metadata["logprobs"]["content"][0]
        prob = math.exp(logprob["logprob"])
        token = logprob["token"]
        if token == "True":
            return prob
        elif token == "False":
            return 1.0 - prob
        else:
            raise ValueError(f"Unexpected logprob: {logprob}")

    async def extract_claims(self, question: str, communities: Iterable[Iterable[Document]]) -> list[Claim]:
        """Extract claims from document communities.
        
        Args:
            question: The question being answered
            communities: Document communities to extract claims from
            
        Returns:
            list[Claim]: Extracted claims
        """
        claim_chain = (
            RunnableParallel(
                {
                    "question": lambda x: x["question"],
                    "formatted_documents": lambda x: self.format_documents_with_ids(x["documents"]),
                }
            )
            | self.claims_prompt
            | self.claims_model
        )

        community_claims = await claim_chain.abatch(
            [{"question": question, "documents": community} for community in communities]
        )
        return [claim for community in community_claims for claim in community.claims]

    async def rank_claims(self, question: str, claims: Iterable[Claim]) -> list[Claim]:
        """Rank claims by relevance to the question.
        
        Args:
            question: The question being answered
            claims: Claims to rank
            
        Returns:
            list[Claim]: Ranked claims
        """
        rank_chain = self.rank_prompt | self.model.bind(logprobs=True) | RunnableLambda(self.compute_rank)

        ranks = await rank_chain.abatch(
            [{"question": question, "claim": claim.claim} for claim in claims]
        )
        rank_claims = sorted(
            zip(ranks, claims, strict=True), key=lambda rank_claim: rank_claim[0]
        )

        return [claim for _, claim in rank_claims] 