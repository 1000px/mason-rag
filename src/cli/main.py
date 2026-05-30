import argparse
from pathlib import Path

from src.core.ingestion import IngestionPipeline


def cmd_ingest(args):
    pipeline = IngestionPipeline()
    result = pipeline.ingest(args.file_path)
    print(f"\n✅ 摄取完成！")
    print(f"   文件: {result['file_name']}")
    print(f"   段落数: {result['document_count']}")
    print(f"   分块数: {result['chunk_count']}")
    print(f"   向量数: {result['vector_count']}")


def cmd_search(args):
    pipeline = IngestionPipeline()
    docs = pipeline.search(args.query, top_k=args.top_k)
    if not docs:
        print("❌ 未找到相关内容")
        return
    print(f"\n🔍 搜索结果 (Top {len(docs)}):")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", "未知")
        content = doc.page_content[:200].replace("\n", " ")
        print(f"\n--- [{i}] 来源: {source} ---")
        print(f"    {content}...")


def cmd_list(args):
    pipeline = IngestionPipeline()
    files = pipeline.list_files()
    if not files:
        print("📭 向量库中没有文档")
        return
    print(f"\n📚 已索引的文档 ({len(files)} 个):")
    for f in files:
        print(f"   • {f}")


def cmd_delete(args):
    pipeline = IngestionPipeline()
    count = pipeline.delete_file(args.file_name)


def main():
    parser = argparse.ArgumentParser(
        description="Mason-RAG 文档摄取工具",
        prog="mason-rag",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    ingest_parser = subparsers.add_parser("ingest", help="摄取文档到向量库")
    ingest_parser.add_argument("file_path", type=str, help="文档路径 (pdf/txt/md)")

    search_parser = subparsers.add_parser("search", help="搜索文档")
    search_parser.add_argument("query", type=str, help="搜索关键词")
    search_parser.add_argument("-k", "--top-k", type=int, default=3, help="返回结果数 (默认: 3)")

    subparsers.add_parser("list", help="列出已索引的文档")

    delete_parser = subparsers.add_parser("delete", help="删除文档")
    delete_parser.add_argument("file_name", type=str, help="文件名 (如: 制度.pdf)")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()