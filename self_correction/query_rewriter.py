"""
QueryRewriter - Improves retrieval by expanding and refining user queries

Strategies:
- Generate multiple query variations (synonyms, rephrasing)
- Refine queries based on failed retrieval attempts
- Extract key concepts from queries
"""
import ollama
from typing import List


class QueryRewriter:
    """Rewrites and expands queries for better retrieval"""
    
    def __init__(self, llm_model: str = "llama3"):
        self.llm_model = llm_model
    
    def rewrite(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple query variations for better retrieval.
        
        Returns list starting with original query, followed by variations.
        """
        print(f"🔄 [QueryRewriter] Generating {num_variations} variations...")
        
        prompt = f"""Given this user question, generate {num_variations} alternative versions that would help retrieve relevant information from a document database. Focus on:
- Synonyms and related terms
- Different phrasings of the same intent
- More specific or more general versions

Original question: "{query}"

Output ONLY the alternative queries, one per line, numbered 1-{num_variations}. No explanations."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Parse the response into individual queries
            raw_output = response['message']['content']
            variations = self._parse_variations(raw_output)
            
            # Always include original query first
            all_queries = [query] + variations[:num_variations]
            
            print(f"✅ [QueryRewriter] Generated {len(all_queries)} total queries")
            return all_queries
            
        except Exception as e:
            print(f"⚠️ [QueryRewriter] Error: {e}, using original query only")
            return [query]
    
    def refine_with_feedback(self, original_query: str, failed_context: str) -> str:
        """
        Refine query based on poor retrieval results.
        Used when initial retrieval returns irrelevant context.
        """
        print("🔧 [QueryRewriter] Refining query based on feedback...")
        
        prompt = f"""The following query was used to search a document database, but the retrieved context wasn't helpful.

Original Query: "{original_query}"

Retrieved Context (not helpful):
{failed_context[:500]}...

Based on this, suggest ONE improved query that would find more relevant information. The query should:
- Be more specific or use different terminology
- Target the actual information needed
- Avoid terms that led to irrelevant results

Output ONLY the improved query, nothing else."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            refined = response['message']['content'].strip()
            # Clean up any quotes or extra formatting
            refined = refined.strip('"\'')
            
            print(f"✅ [QueryRewriter] Refined to: {refined[:100]}...")
            return refined
            
        except Exception as e:
            print(f"⚠️ [QueryRewriter] Refinement failed: {e}")
            return original_query
    
    def extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts/entities from query for targeted search"""
        print("🔍 [QueryRewriter] Extracting key concepts...")
        
        prompt = f"""Extract the key concepts, entities, and important terms from this question. These will be used for document search.

Question: "{query}"

Output only the key terms, one per line. No explanations."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            concepts = [
                line.strip().strip('-•*')
                for line in response['message']['content'].split('\n')
                if line.strip()
            ]
            
            return concepts[:10]  # Limit to 10 concepts
            
        except Exception as e:
            print(f"⚠️ [QueryRewriter] Concept extraction failed: {e}")
            return [query]
    
    def _parse_variations(self, raw_output: str) -> List[str]:
        """Parse LLM output into list of query variations"""
        variations = []
        
        for line in raw_output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Remove common prefixes: "1.", "1)", "-", "*", etc.
            import re
            cleaned = re.sub(r'^[\d]+[.\)]\s*', '', line)
            cleaned = cleaned.strip('-•*').strip()
            cleaned = cleaned.strip('"\'')
            
            if len(cleaned) > 5:  # Minimum viable query length
                variations.append(cleaned)
        
        return variations
