"""
AnswerValidator - Validates generated answers against source context

Strategies:
- Check if answer is grounded in retrieved context
- Detect potential hallucinations
- Score confidence based on context coverage
"""
import ollama
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of answer validation"""
    confidence: float  # 0.0 - 1.0
    is_grounded: bool  # Answer is supported by context
    unsupported_claims: List[str]  # Statements not in context
    reasoning: str  # Explanation of validation


@dataclass 
class HallucinationReport:
    """Report on detected hallucinations"""
    has_hallucinations: bool
    detected_items: List[str]  # Potentially hallucinated statements
    severity: str  # 'none', 'minor', 'major'
    suggested_fix: str  # How to correct the answer


class AnswerValidator:
    """Validates answers against source context to ensure factual accuracy"""
    
    def __init__(self, llm_model: str = "llama3"):
        self.llm_model = llm_model
    
    def validate(self, answer: str, context: List[str], query: str) -> ValidationResult:
        """
        Validate that the answer is grounded in the provided context.
        
        Returns ValidationResult with confidence score and grounding analysis.
        """
        print("🔍 [Validator] Checking answer grounding...")
        
        context_text = "\n\n---\n\n".join(context)
        
        prompt = f"""You are a fact-checker. Analyze if the given answer is supported by the provided context.

CONTEXT:
{context_text[:3000]}

QUESTION: {query}

ANSWER: {answer}

Analyze the answer and respond in this exact format:
CONFIDENCE: [0.0-1.0 how well the answer is supported]
GROUNDED: [YES/NO]
UNSUPPORTED: [List any claims not found in context, or "none"]
REASONING: [Brief explanation]"""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            result = self._parse_validation(response['message']['content'])
            print(f"✅ [Validator] Confidence: {result.confidence:.2f}")
            return result
            
        except Exception as e:
            print(f"⚠️ [Validator] Error: {e}")
            return ValidationResult(
                confidence=0.5,
                is_grounded=True,
                unsupported_claims=[],
                reasoning=f"Validation failed: {e}"
            )
    
    def check_hallucination(self, answer: str, context: List[str]) -> HallucinationReport:
        """
        Detect statements that appear fabricated or not supported by context.
        Uses prompt-based detection (simpler than NLI model).
        """
        print("🔎 [Validator] Checking for hallucinations...")
        
        context_text = "\n\n---\n\n".join(context)
        
        prompt = f"""You are a hallucination detector. Identify any statements in the answer that are NOT supported by the context.

CONTEXT:
{context_text[:3000]}

ANSWER TO CHECK:
{answer}

Identify fabricated or unsupported facts. Respond in this exact format:
HAS_HALLUCINATIONS: [YES/NO]
SEVERITY: [none/minor/major]
DETECTED: [List specific hallucinated statements, or "none"]
FIX: [How to correct the answer, or "No fix needed"]"""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            result = self._parse_hallucination(response['message']['content'])
            
            if result.has_hallucinations:
                print(f"⚠️ [Validator] Hallucinations detected: {result.severity}")
            else:
                print("✅ [Validator] No hallucinations detected")
                
            return result
            
        except Exception as e:
            print(f"⚠️ [Validator] Hallucination check failed: {e}")
            return HallucinationReport(
                has_hallucinations=False,
                detected_items=[],
                severity='none',
                suggested_fix="Check failed"
            )
    
    def score_relevance(self, context: List[str], query: str) -> float:
        """
        Quick check if retrieved context is relevant to the query.
        Returns score 0.0-1.0
        """
        if not context:
            return 0.0
        
        prompt = f"""Rate how relevant this context is for answering the question.

QUESTION: {query}

CONTEXT (first 500 chars):
{context[0][:500]}...

Reply with ONLY a number from 0 to 10 (10 = perfectly relevant)."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Extract number from response
            import re
            match = re.search(r'\d+', response['message']['content'])
            if match:
                score = min(10, max(0, int(match.group()))) / 10.0
                return score
            return 0.5
            
        except Exception:
            return 0.5
    
    def _parse_validation(self, raw: str) -> ValidationResult:
        """Parse LLM validation response"""
        lines = raw.strip().split('\n')
        
        confidence = 0.5
        is_grounded = True
        unsupported = []
        reasoning = ""
        
        for line in lines:
            line_upper = line.upper()
            if 'CONFIDENCE:' in line_upper:
                try:
                    # Extract number
                    import re
                    match = re.search(r'[\d.]+', line)
                    if match:
                        confidence = float(match.group())
                        if confidence > 1:
                            confidence = confidence / 10  # Handle 0-10 scale
                except:
                    pass
                    
            elif 'GROUNDED:' in line_upper:
                is_grounded = 'YES' in line_upper
                
            elif 'UNSUPPORTED:' in line_upper:
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if content.lower() not in ['none', 'n/a', '']:
                    unsupported = [s.strip() for s in content.split(',')]
                    
            elif 'REASONING:' in line_upper:
                reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
        
        return ValidationResult(
            confidence=confidence,
            is_grounded=is_grounded,
            unsupported_claims=unsupported,
            reasoning=reasoning
        )
    
    def _parse_hallucination(self, raw: str) -> HallucinationReport:
        """Parse LLM hallucination check response"""
        lines = raw.strip().split('\n')
        
        has_halluc = False
        severity = 'none'
        detected = []
        fix = ""
        
        for line in lines:
            line_upper = line.upper()
            if 'HAS_HALLUCINATIONS:' in line_upper:
                has_halluc = 'YES' in line_upper
                
            elif 'SEVERITY:' in line_upper:
                content = line.split(':', 1)[1].strip().lower() if ':' in line else ""
                if content in ['minor', 'major', 'none']:
                    severity = content
                    
            elif 'DETECTED:' in line_upper:
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if content.lower() not in ['none', 'n/a', '']:
                    detected = [s.strip() for s in content.split(',') if s.strip()]
                    
            elif 'FIX:' in line_upper:
                fix = line.split(':', 1)[1].strip() if ':' in line else ""
        
        return HallucinationReport(
            has_hallucinations=has_halluc,
            detected_items=detected,
            severity=severity,
            suggested_fix=fix
        )
