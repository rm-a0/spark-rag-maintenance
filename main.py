"""
main.py

Usage:
    python main.py --pipeline
    python main.py --train
    python main.py --predict --engine 42
    python main.py --predict --engine 42 --no-rag
    python main.py --pipeline --train
"""
import argparse
import sys

from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance")

    parser.add_argument("--pipeline", action="store_true", help="Run Spark feature pipeline")
    parser.add_argument("--train",    action="store_true", help="Train XGBoost models")
    parser.add_argument("--predict",  action="store_true", help="Predict risk for an engine")
    parser.add_argument("--engine",   type=int,            help="Engine ID to diagnose")
    parser.add_argument("--no-rag",   action="store_true", help="Skip RAG diagnostic")

    args = parser.parse_args()

    if not any([args.pipeline, args.train, args.predict]):
        parser.print_help()
        sys.exit(1)

    if args.pipeline:
        from src.pipeline import run_pipeline
        run_pipeline()

    if args.train:
        if not settings.parquet_path.exists():
            print(f"Parquet not found at {settings.parquet_path}")
            print("Run: python main.py --pipeline first")
            sys.exit(1)
        from src.train import run_training
        run_training()

    if args.predict:
        if args.engine is None:
            print("--predict requires --engine <id>")
            sys.exit(1)
        from src.predict import run_predict
        report = run_predict(args.engine, use_rag=not args.no_rag)
        report.print()


if __name__ == "__main__":
    main()