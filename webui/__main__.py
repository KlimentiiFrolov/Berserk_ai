import argparse
from webui.demo import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio UI для RAG")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="публичная ссылка gradio.live")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()