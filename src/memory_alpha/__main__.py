import asyncio
import argparse

from memory_alpha.server import mcp
from memory_alpha.settings import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Memory Alpha Server")
    parser.add_argument(
        "--mode",
        choices=["stdio", "sse"],
        default=settings.server_mode,
        help="Server mode: stdio (default) or sse",
    )
    parser.add_argument(
        "--host",
        default=settings.server_host,
        help=f"Host to bind to in SSE mode (default: {settings.server_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.server_port,
        help=f"Port to listen on in SSE mode (default: {settings.server_port})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "sse":
        print(f"Starting server in SSE mode on {args.host}:{args.port}")
        asyncio.run(mcp.run_sse_async(host=args.host, port=args.port))
    else:
        # print("Starting server in stdio mode")
        mcp.run()


if __name__ == "__main__":
    main()
