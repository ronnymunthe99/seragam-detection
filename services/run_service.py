import os, sys
import argparse

path_this = os.path.realpath(os.path.dirname(__file__))
path_project = os.path.realpath(os.path.join(path_this, ".."))
project_root = os.path.abspath(os.path.join(path_this, "../../"))
sys.path.append(path_this)
sys.path.append(path_project)
sys.path.append(project_root)

import uvicorn
from uvicorn.config import LOGGING_CONFIG

# python run_service.py -p 6969 -w 0

def get_args():
    parser = argparse.ArgumentParser(
        description='LLaVA API'
    )

    parser.add_argument(
        '-p', '--port',
        help='Port to listen on',
        type=int,
        default=5000
    )

    parser.add_argument(
        '-H', '--host',
        help='Host to bind to',
        default='0.0.0.0'
    )
    
    parser.add_argument(
        "-w", "--workers",
        help="Number of workers",
        type=int,
        default=0
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s  %(message)s"
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
    uvicorn.run("service:app", host=args.host, port=args.port, workers = args.workers, log_level="info", reload=False)
