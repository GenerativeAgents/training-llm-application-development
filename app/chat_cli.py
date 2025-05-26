"""
会話履歴を蓄積しながら対話を行うCLIアプリケーション

実行方法:
    uv run python app/chat_cli.py

停止方法:
    Ctrl + C
"""

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def generate(messages: list[ChatCompletionMessageParam]) -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
    )
    return response.choices[0].message.content  # type: ignore[return-value]


def main() -> None:
    load_dotenv(override=True)

    # 会話履歴を初期化
    messages: list[ChatCompletionMessageParam] = [
        {"role": "developer", "content": "You are a helpful assistant."},
    ]

    while True:
        # ユーザーの入力を受け付けて会話履歴に追加
        user_message = input("User: ")
        messages.append({"role": "user", "content": user_message})

        # LLMによる応答を生成して会話履歴に追加
        ai_message = generate(messages=messages)
        messages.append({"role": "assistant", "content": ai_message})
        print(f"Assistant: {ai_message}")


if __name__ == "__main__":
    main()
