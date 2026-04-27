"""
FairForge Gemini Auditor — Uses Gemini 1.5 Flash to generate human-readable
bias audit explanations and what-if counterfactuals
"""
import google.generativeai as genai
import os
from typing import Optional
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_audit_narrative(report_dict: dict, domain: str = "hiring") -> str:
    """
    Takes a FairnessReport dict, returns plain-English audit summary.
    """
    prompt = f"""
You are a fairness auditor reviewing an AI model used in {domain} decisions.
Here are the fairness metrics:

{report_dict}

Write a clear, professional 3-paragraph audit report for a non-technical executive:
1. What biases were detected and how severe they are
2. Which groups are most affected and what the real-world impact is
3. Top 2 recommended fixes

Use plain language. Be specific about numbers. Be direct about severity.
"""
    response = model.generate_content(prompt)
    return response.text


def generate_counterfactual_explanation(
    individual: dict,
    decision: str,
    sensitive_attr: str,
    counterfactual_attr_value: str
) -> str:
    """
    Explains what would happen if only the protected attribute changed.
    """
    prompt = f"""
An AI made this decision: {decision}
For this person: {individual}

If only the {sensitive_attr} was changed to {counterfactual_attr_value} 
(keeping everything else identical), what would likely happen? 
Why does this indicate bias? Explain in 2-3 sentences for a general audience.
"""
    response = model.generate_content(prompt)
    return response.text


def suggest_policy_fix(policy_id: str, violation_details: dict) -> str:
    """Uses Gemini to explain a specific policy violation in plain English."""
    prompt = f"""
A fairness policy {policy_id} was violated. Details: {violation_details}
Explain this violation and its legal/ethical implications in 2 clear sentences.
Then give one concrete technical fix.
"""
    response = model.generate_content(prompt)
    return response.text