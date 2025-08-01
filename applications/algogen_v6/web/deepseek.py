import httpx
from volcenginesdkarkruntime import Ark

client = Ark(    
    api_key="5ab0d1a0-f268-466c-99b8-9ee39685e334",
    # The output time of the reasoning model is relatively long. Please increase the timeout period.
    timeout=httpx.Timeout(timeout=1800),
)
if __name__ == "__main__":
    # [Recommended] Streaming:
    print("----- streaming request -----")
    stream = client.chat.completions.create(
        model="deepseek-r1-250528",
        messages=[
            {"role": "user", "content": "常见的十字花科植物有哪些？"},
        ],
        stream=True
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.reasoning_content:
            print(chunk.choices[0].delta.reasoning_content, end="")
        else:
            print(chunk.choices[0].delta.content, end="")
    print()
    # Non-streaming:
    print("----- standard request -----")
    completion = client.chat.completions.create(
        model="deepseek-r1-250528",
        messages=[
            {"role": "user", "content": "常见的十字花科植物有哪些？"},
        ],
    )
    print(completion.choices[0].message.reasoning_content)
    print(completion.choices[0].message.content)