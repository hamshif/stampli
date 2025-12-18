
import requests
import json
import time

API_URL = "http://127.0.0.1:8000/api/stampli"

def interact(messages):
    print(f"\n--- Sending {len(messages)} messages ---")
    print(f"Last User: {messages[-1]['content']}")
    
    payload = {"messages": messages}
    full_response = ""
    
    with requests.post(API_URL, json=payload, stream=True) as r:
        if r.status_code != 200:
            print(f"Error: {r.status_code}")
            return ""
            
        for line in r.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str == "[DONE]": break
                    try:
                        data = json.loads(data_str)
                        if data['type'] == 'log':
                            print(f"[LOG] {data['data']['text']}")
                        elif data['type'] == 'delta':
                            full_response += data['data']['text']
                    except: pass
    
    print(f"\n[AI]: {full_response.strip()}")
    return full_response

def main():
    msgs = []
    
    # 1. Query that supposedly returns "Cannot answer" despite matches
    q1 = "What do visitors from Australia say about Disneyland in Paris?"
    msgs.append({"role": "user", "content": q1})
    resp1 = interact(msgs)
    msgs.append({"role": "assistant", "content": resp1})
    
    # 2. Context test
    q2 = "Why?"
    msgs.append({"role": "user", "content": q2})
    interact(msgs)

if __name__ == "__main__":
    main()
