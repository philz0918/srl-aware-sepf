import os

from detector import predict_politeness
from explainer import create_client, explain_with_gpt_oss_from_result


CKPT_FILE = os.getenv(
    "POLITENESS_CKPT_FILE",
    "PATH_for_best_model_4_directional.ckpt",
)
GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "gpt-oss-120b")


def main():
    client = create_client(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    sentence = "I told you I should go"
    result = predict_politeness(sentence, ckpt_file=CKPT_FILE)
    llm_out = explain_with_gpt_oss_from_result(client, result, model_name=GPT_OSS_MODEL)

    print("Score:", result["predicted_politeness"])
    print("SRL descriptions:")
    for desc in result["srl_descriptions"]:
        print("-", desc)

    print("\nFrame summary:")
    print(llm_out["frame_summary"])
    print("\nExplanation:")
    print(llm_out["explanation"])


if __name__ == "__main__":
    main()
