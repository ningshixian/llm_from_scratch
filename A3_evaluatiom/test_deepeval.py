from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval

"""
evaluate LLM outputs using metrics like hallucination, relevancy, and faithfulness.
"""

def test_professionalism():
    # Define the metric
    metric = GEval(
        name="Professionalism",
        criteria="Determine if the response is professional and helpful.",
        threshold=0.7
    )
    
    # Define the test case
    test_case = LLMTestCase(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking 'Forgot Password' on the login page."
    )
    
    # Execute the test
    assert_test(test_case, [metric])
