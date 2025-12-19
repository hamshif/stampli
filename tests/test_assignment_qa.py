import requests
import json
import sys
import time
import subprocess
import os

# Configuration
API_URL = "http://127.0.0.1:8000/api/stampli"

# The North Star Questions + Expanded Suite
TEST_CASES = [
    # --- 1. Core Assignment Questions ---
    {
        "name": "Q1: Australia -> HongKong",
        "question": "What do visitors from Australia say about Disneyland in HongKong?",
        "expected_keywords": ["Australia", "Hong Kong"],
        "forbidden_keywords": ["cannot answer", "no specific filters"]
    },
    {
        "name": "Q2: Spring (Season)",
        "question": "Is spring a good time to visit Disneyland in HongKong?",
        "expected_keywords": ["Spring", "March", "April", "May", "weather", "Hong Kong"],
        "forbidden_keywords": ["cannot answer"]
    },
    {
        "name": "Q3: California Crowd (Enrichment)",
        "question": "Is Disneyland California usually crowded in June?",
        "expected_keywords": ["June", "California", "crowd", "wait", "packed", "busy"],
        "forbidden_keywords": ["cannot answer"]
    },
    {
        "name": "Q4: Paris Staff (Enrichment)",
        "question": "Is the staff in Paris friendly?",
        "expected_keywords": ["Paris", "staff", "friendly", "rude"],
        "forbidden_keywords": ["cannot answer"]
    },

    # --- 2. Enrichment Backfill Stress Tests ---
    {
        "name": "Q5: California Price (Sensitivity)",
        "question": "Is the food expensive in Disneyland California?",
        "expected_keywords": ["California", "food", "price", "expensive", "money"],
        "forbidden_keywords": ["cannot answer"]
    },
    {
        "name": "Q6: Paris Crowd (Packed check)",
        "question": "Are the lines long in Disneyland Paris?",
        "expected_keywords": ["Paris", "lines", "queues", "wait", "crowd"],
        "forbidden_keywords": ["cannot answer"]
    },
    {
        "name": "Q7: HongKong Family (Sentiment)",
        "question": "Is Disneyland Hong Kong good for families?",
        "expected_keywords": ["Hong Kong", "kids", "family", "children"],
        "forbidden_keywords": ["cannot answer"]
    },
    
    # --- 3. Follow-up / Context Tests ---
    {
        "name": "Q8: Follow-up (Why?)",
        "question": "Why?",
        "context_required": True, # Should follow Q7
        "expected_keywords": [],
        "forbidden_keywords": ["cannot answer"]
    },
    
    # --- 4. Negative / Guardrail Tests ---
    {
        "name": "Q9: Invalid Entity",
        "question": "Is the weather good in Disneyland Tel Aviv?",
        "expected_keywords": ["not found", "available", "California", "Paris", "Hong Kong"], 
        "forbidden_keywords": [] # Actually expect a polite refusal
    }
]

def run_test(case, session=None):
    print(f"\n==================================================")
    print(f"TEST: {case['name']}")
    print(f"Q: {case['question']}")
    print(f"==================================================")
    
    messages = session if session else []
    messages.append({"role": "user", "content": case['question']})
    
    payload = {"messages": messages}
    start = time.time()
    
    try:
        full_response = ""
        print("[Streaming] ", end="")
        with requests.post(API_URL, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if data['type'] == 'log':
                                # print(f"  [LOG] {data['data']['text']}")
                                pass
                            elif data['type'] == 'delta':
                                text = data['data']['text']
                                print(text, end="", flush=True)
                                full_response += text
                        except:
                            pass
        print("\n")
        
        duration = time.time() - start
        print(f"[Time]: {duration:.2f}s")
        
        # Assertions
        passed = True
        
        # Check Forbidden
        for bad in case.get('forbidden_keywords', []):
            if bad.lower() in full_response.lower():
                print(f"❌ FAILED: Found forbidden keyword '{bad}'")
                passed = False
                
        # Check Expected
        lower_resp = full_response.lower()
        if case.get('expected_keywords'):
            hits = [k for k in case['expected_keywords'] if k.lower() in lower_resp]
            if not hits:
                print(f"❌ FAILED: Missed all expected keywords: {case['expected_keywords']}")
                passed = False
            else:
                print(f"✅ Matched keywords: {hits}")

        # Length Check
        if len(full_response) < 20:
             print("❌ FAILED: Response too short")
             passed = False
             
        if passed:
            # Emoji Check
            # Basic ranges for emojis (surrogates, symbols, etc.)
            # This is a heuristic.
            if any(char in full_response for char in "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​"): 
                # Ideally we use regex for range, but without extra deps, let's keep it simple or just rely on manual inspection if this is too hard.
                # Actually, let's use a regex for high unicode characters if possible, or just checks specific ones.
                # Re-thinking: The user wants to INSTRUCT the model. Verification is best effort.
                pass

            # Check for non-ascii characters that are likely emojis
            # Emoji range usually starts high.
            # Simple check: if there are characters outside basic multilingual plane (BMP) or specific symbol blocks.
            for char in full_response:
                # High surrogate pairs or specific symbol ranges often indicate emojis
                if ord(char) > 0x1F600 and ord(char) < 0x1F650: # Emoticons
                     print(f"❌ FAILED: Found emoji {char}")
                     passed = False
                     break
            
            if passed:
                print("STATUS: ✅ PASSED")
            
                # Return updated session for context tests
                messages.append({"role": "assistant", "content": full_response})
                return True, messages
            else:
                 print("STATUS: ❌ FAILED (Emoji Detected)")
                 return False, messages
        else:
            print("STATUS: ❌ FAILED")
            return False, messages

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, messages

def main():
    # Wait for server
    print("Checking server health...")
    try:
        requests.get("http://127.0.0.1:8000/docs", timeout=2)
        print("Server is UP.")
    except:
        print("Server is DOWN. Please start stampli_chat.py")
        sys.exit(1)

    # Run Tests
    passed_count = 0
    
    current_session = []
    
    for case in TEST_CASES:
        if not case.get('context_required'):
            current_session = []
            
        is_pass, new_session = run_test(case, current_session)
        if is_pass:
            passed_count += 1
        
        current_session = new_session
        time.sleep(0.5)
        
    print(f"\n\nTest Suite Completed: {passed_count}/{len(TEST_CASES)} Passed")
    if passed_count < len(TEST_CASES):
        sys.exit(1)

if __name__ == "__main__":
    main()
