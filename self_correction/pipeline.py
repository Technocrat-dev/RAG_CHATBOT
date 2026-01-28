"""
SelfCorrectingRAG - Complete self-correcting RAG pipeline

Combines QueryRewriter and AnswerValidator for iterative refinement:
1. Rewrite query for better retrieval
2. Retrieve context from vector store (hybrid search)
3. Re-rank retrieved contexts for precision
4. Generate answer
5. Validate and check for hallucinations
6. If validation fails, refine and retry
"""
import ollama
from typing import List, Optional, Generator
from dataclasses import dataclass

from .query_rewriter import QueryRewriter
from .validator import AnswerValidator, ValidationResult, HallucinationReport
from search.reranker import Reranker


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
    2. Retrieve context for all query variations (hybrid BM25 + dense)
    3. Re-rank contexts using cross-encoder
    4. Generate initial answer
    5. Validate answer against context
    6. If validation fails (low confidence), refine query and retry
    7. Check for hallucinations and remove if found
    8. Return final answer with confidence score
    """
    
    def __init__(
        self,
        db,  # VectorDB instance
        llm_model: str = "llama3",
        max_iterations: int = 2,
        confidence_threshold: float = 0.6,
        use_reranker: bool = True
    ):
        self.db = db
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        
        self.rewriter = QueryRewriter(llm_model)
        self.validator = AnswerValidator(llm_model)
        
        # Optional re-ranker for improved precision
        self.reranker = Reranker() if use_reranker else None
    
    def query(self, user_query: str, collection_id: str = None) -> RAGResponse:
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
            
            # Step 2: Retrieve context for all query variations (over-retrieve for re-ranking)
            all_contexts = []
            seen_texts = set()
            
            # Retrieve more candidates if using re-ranker
            retrieve_top_k = 8 if self.reranker else 4
            
            for q in queries:
                contexts = self.db.retrieve(q, top_k=retrieve_top_k, collection_id=collection_id)
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
            
            # Step 3: Re-rank for precision (if enabled)
            if self.reranker and len(all_contexts) > 4:
                print("🔄 Re-ranking contexts for precision...")
                all_contexts = self.reranker.rerank(user_query, all_contexts, top_k=4)
                print(f"✅ Selected top {len(all_contexts)} after re-ranking")
            
            # Step 4: Check context relevance
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
    
    def query_streaming(self, user_query: str, collection_id: str = None) -> Generator[dict, None, None]:
        """
        Execute self-correcting pipeline with streaming log events.
        
        Yields:
            dict: Event objects with types: 'log', 'token', 'done'
        """
        yield {"type": "log", "step": "start", "message": f"Processing query: {user_query[:50]}..."}
        
        best_answer = None
        best_confidence = 0.0
        best_contexts = []
        was_corrected = False
        current_query = user_query
        
        for iteration in range(self.max_iterations):
            yield {"type": "log", "step": "iteration", "message": f"Iteration {iteration + 1}/{self.max_iterations}"}
            
            # Step 1: Expand query
            if iteration == 0:
                yield {"type": "log", "step": "rewrite", "message": "Expanding query into variations..."}
                queries = self.rewriter.rewrite(current_query, num_variations=2)
                yield {"type": "log", "step": "rewrite", "message": f"Generated {len(queries)} query variations"}
            else:
                queries = [current_query]
            
            # Step 2: Retrieve context
            yield {"type": "log", "step": "retrieve", "message": "Retrieving context chunks..."}
            all_contexts = []
            seen_texts = set()
            retrieve_top_k = 8 if self.reranker else 4
            
            for q in queries:
                contexts = self.db.retrieve(q, top_k=retrieve_top_k, collection_id=collection_id)
                for ctx in contexts:
                    ctx_hash = ctx[:100]
                    if ctx_hash not in seen_texts:
                        all_contexts.append(ctx)
                        seen_texts.add(ctx_hash)
            
            if not all_contexts:
                yield {"type": "log", "step": "retrieve", "message": "⚠️ No context found", "warning": True}
                continue
            
            yield {"type": "log", "step": "retrieve", "message": f"Retrieved {len(all_contexts)} unique chunks"}
            
            # Step 3: Re-rank
            if self.reranker and len(all_contexts) > 4:
                yield {"type": "log", "step": "rerank", "message": "Re-ranking contexts for precision..."}
                all_contexts = self.reranker.rerank(user_query, all_contexts, top_k=4)
                yield {"type": "log", "step": "rerank", "message": f"Selected top {len(all_contexts)} contexts"}
            
            # Step 4: Check relevance
            yield {"type": "log", "step": "relevance", "message": "Checking context relevance..."}
            relevance = self.validator.score_relevance(all_contexts, user_query)
            yield {"type": "log", "step": "relevance", "message": f"Relevance score: {relevance:.2f}"}
            
            if relevance < 0.3 and iteration < self.max_iterations - 1:
                yield {"type": "log", "step": "refine", "message": "Context irrelevant, refining query...", "warning": True}
                current_query = self.rewriter.refine_with_feedback(user_query, all_contexts[0] if all_contexts else "")
                continue
            
            # Step 5: Generate answer
            yield {"type": "log", "step": "generate", "message": "Generating answer..."}
            answer = self._generate_answer(user_query, all_contexts)
            
            # Step 6: Validate
            yield {"type": "log", "step": "validate", "message": "Validating answer..."}
            validation = self.validator.validate(answer, all_contexts, user_query)
            yield {"type": "log", "step": "validate", "message": f"Confidence: {validation.confidence:.0%}"}
            
            if validation.confidence > best_confidence:
                best_answer = answer
                best_confidence = validation.confidence
                best_contexts = all_contexts
            
            # Step 7: Check hallucinations if good enough
            if validation.confidence >= self.confidence_threshold:
                yield {"type": "log", "step": "hallucination", "message": "Checking for hallucinations..."}
                halluc = self.validator.check_hallucination(answer, all_contexts)
                
                if halluc.has_hallucinations and halluc.severity != 'none':
                    yield {"type": "log", "step": "correct", "message": "Correcting hallucinations...", "warning": True}
                    answer = self._correct_hallucinations(answer, halluc, all_contexts)
                    was_corrected = True
                else:
                    yield {"type": "log", "step": "hallucination", "message": "No hallucinations detected ✓"}
                
                # Done! Yield final result
                yield {
                    "type": "done",
                    "answer": answer,
                    "confidence": validation.confidence,
                    "sources": all_contexts[:3],
                    "iterations": iteration + 1,
                    "was_corrected": was_corrected
                }
                return
            
            # Step 8: Refine for next iteration
            if iteration < self.max_iterations - 1:
                yield {"type": "log", "step": "refine", "message": f"Confidence {validation.confidence:.0%} below threshold, refining..."}
                current_query = self.rewriter.refine_with_feedback(user_query, "\n".join(all_contexts[:2]))
                was_corrected = True
        
        # Return best answer after all iterations
        yield {"type": "log", "step": "complete", "message": f"Completed with best confidence: {best_confidence:.0%}"}
        yield {
            "type": "done",
            "answer": best_answer or "I couldn't find a reliable answer in the documents.",
            "confidence": best_confidence,
            "sources": best_contexts[:3],
            "iterations": self.max_iterations,
            "was_corrected": was_corrected
        }

