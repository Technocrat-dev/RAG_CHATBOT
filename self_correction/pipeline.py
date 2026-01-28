"""
SelfCorrectingRAG - Complete self-correcting RAG pipeline

Combines QueryRewriter and AnswerValidator for iterative refinement:
1. Rewrite query for better retrieval
2. Retrieve context from vector store
3. Generate answer
4. Validate and check for hallucinations
5. If validation fails, refine and retry
"""
import ollama
from typing import List, Optional
from dataclasses import dataclass

from .query_rewriter import QueryRewriter
from .validator import AnswerValidator, ValidationResult, HallucinationReport


@dataclass
class RAGResponse:
    """Complete response from self-correcting RAG pipeline"""
    answer: str
    sources: List[str]  # Context chunks used
    confidence: float
    iterations: int  # How many attempts were needed
    was_corrected: bool  # Whether answer was modified
    validation: Optional[ValidationResult] = None


class SelfCorrectingRAG:
    """
    Self-correcting RAG pipeline that validates and improves responses.
    
    Workflow:
    1. Expand query into multiple variations
    2. Retrieve context for all query variations
    3. Generate initial answer
    4. Validate answer against context
    5. If validation fails (low confidence), refine query and retry
    6. Check for hallucinations and remove if found
    7. Return final answer with confidence score
    """
    
    def __init__(
        self,
        db,  # VectorDB instance
        llm_model: str = "llama3",
        max_iterations: int = 2,
        confidence_threshold: float = 0.6
    ):
        self.db = db
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        
        self.rewriter = QueryRewriter(llm_model)
        self.validator = AnswerValidator(llm_model)
    
    def query(self, user_query: str) -> RAGResponse:
        """
        Execute the self-correcting RAG pipeline.
        
        Returns RAGResponse with answer, sources, and metadata.
        """
        print(f"\n{'='*60}")
        print(f"🚀 [SelfCorrectingRAG] Processing: {user_query[:50]}...")
        print(f"{'='*60}")
        
        best_answer = None
        best_confidence = 0.0
        best_contexts = []
        best_validation = None
        was_corrected = False
        
        # Track all queries tried
        current_query = user_query
        
        for iteration in range(self.max_iterations):
            print(f"\n📍 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Expand query into variations
            if iteration == 0:
                queries = self.rewriter.rewrite(current_query, num_variations=2)
            else:
                # For retries, use refined query
                queries = [current_query]
            
            # Step 2: Retrieve context for all query variations
            all_contexts = []
            seen_texts = set()
            
            for q in queries:
                contexts = self.db.retrieve(q, top_k=4)
                for ctx in contexts:
                    # Deduplicate
                    ctx_hash = ctx[:100]
                    if ctx_hash not in seen_texts:
                        all_contexts.append(ctx)
                        seen_texts.add(ctx_hash)
            
            if not all_contexts:
                print("⚠️ No context retrieved")
                continue
            
            print(f"📚 Retrieved {len(all_contexts)} unique context chunks")
            
            # Step 3: Check context relevance
            relevance = self.validator.score_relevance(all_contexts, user_query)
            print(f"📊 Context relevance score: {relevance:.2f}")
            
            if relevance < 0.3 and iteration < self.max_iterations - 1:
                print("⚠️ Context seems irrelevant, refining query...")
                current_query = self.rewriter.refine_with_feedback(
                    user_query, 
                    all_contexts[0] if all_contexts else ""
                )
                continue
            
            # Step 4: Generate answer
            answer = self._generate_answer(user_query, all_contexts)
            
            # Step 5: Validate answer
            validation = self.validator.validate(answer, all_contexts, user_query)
            
            # Track best result
            if validation.confidence > best_confidence:
                best_answer = answer
                best_confidence = validation.confidence
                best_contexts = all_contexts
                best_validation = validation
            
            # Step 6: Check if good enough
            if validation.confidence >= self.confidence_threshold:
                print(f"✅ Confidence {validation.confidence:.2f} >= threshold {self.confidence_threshold}")
                
                # Step 7: Check for hallucinations
                halluc = self.validator.check_hallucination(answer, all_contexts)
                
                if halluc.has_hallucinations and halluc.severity != 'none':
                    print("🔧 Correcting hallucinations...")
                    answer = self._correct_hallucinations(answer, halluc, all_contexts)
                    was_corrected = True
                
                return RAGResponse(
                    answer=answer,
                    sources=all_contexts,
                    confidence=validation.confidence,
                    iterations=iteration + 1,
                    was_corrected=was_corrected,
                    validation=validation
                )
            
            # Step 8: Refine for next iteration
            if iteration < self.max_iterations - 1:
                print(f"⚠️ Confidence {validation.confidence:.2f} < threshold, refining...")
                current_query = self.rewriter.refine_with_feedback(
                    user_query,
                    "\n".join(all_contexts[:2])
                )
                was_corrected = True
        
        # Return best answer after all iterations
        print(f"\n🏁 Returning best answer (confidence: {best_confidence:.2f})")
        
        return RAGResponse(
            answer=best_answer or "I couldn't find a reliable answer in the documents.",
            sources=best_contexts,
            confidence=best_confidence,
            iterations=self.max_iterations,
            was_corrected=was_corrected,
            validation=best_validation
        )
    
    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer from contexts using LLM"""
        print("🤖 Generating answer...")
        
        formatted_context = "\n\n---\n\n".join(contexts[:5])  # Limit to 5 chunks
        
        prompt = f"""You are a helpful assistant. Use ONLY the following context to answer the question.
If the answer isn't clearly in the context, say "Based on the available information, I cannot fully answer this."
Do not make up information.

CONTEXT:
{formatted_context}

QUESTION: {query}

ANSWER:"""

        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    def _correct_hallucinations(
        self, 
        answer: str, 
        halluc: HallucinationReport,
        contexts: List[str]
    ) -> str:
        """Attempt to correct detected hallucinations"""
        
        context_text = "\n\n".join(contexts[:3])
        
        prompt = f"""The following answer contains some unsupported statements. Rewrite it to only include information supported by the context.

ORIGINAL ANSWER:
{answer}

UNSUPPORTED STATEMENTS TO REMOVE:
{', '.join(halluc.detected_items)}

AVAILABLE CONTEXT:
{context_text[:2000]}

Rewrite the answer to be accurate. Only include information from the context."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"⚠️ Correction failed: {e}")
            return answer
