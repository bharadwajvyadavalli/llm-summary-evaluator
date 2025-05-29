"""
Quick Simple Test - Test any PDF documents with any queries
Just run this and it works!
"""


def quick_simple_test():
    """Super simple test - just works with any PDFs"""

    print("🚀 QUICK SIMPLE RAG TEST")
    print("=" * 25)

    # Import components
    from simple_pdf_processor import SimplePDFProcessor
    from vector_store_manager import VectorStoreManager
    from simple_rag_agent import SimpleRAGAgent
    from rag_evaluator import RAGEvaluator

    # Setup
    print("🔧 Setting up...")
    processor = SimplePDFProcessor()
    vector_manager = VectorStoreManager()
    rag_agent = SimpleRAGAgent(vector_manager)
    evaluator = RAGEvaluator()

    # Check if we have documents
    stats = vector_manager.get_collection_stats()
    print(f"📊 Current documents: {stats['total_chunks']} chunks")

    # Process PDFs if needed
    if stats['total_chunks'] == 0:
        print("📚 Processing PDFs...")

        import os
        if not os.path.exists("sample_pdfs"):
            print("❌ No 'sample_pdfs' folder found")
            print("💡 Create 'sample_pdfs/' folder and add any PDF files")
            return

        documents = processor.process_pdfs("sample_pdfs")

        if not documents:
            print("❌ No PDFs could be processed")
            return

        vector_manager.add_documents(documents)
        print(f"✅ Loaded {len(documents)} documents")

    # Test some simple queries
    simple_queries = [
        "What is this document about?",
        "What are the main topics?",
        "What important information is mentioned?"
    ]

    print(f"\n🧪 Testing {len(simple_queries)} simple queries...")

    for i, query in enumerate(simple_queries, 1):
        print(f"\n🔍 Query {i}: {query}")

        try:
            # Get response
            response = rag_agent.query(query, k=2)

            # Quick evaluation
            metrics = evaluator.evaluate_response(response)

            # Show results
            print(f"📝 Answer: {response.response[:100]}...")
            print(f"📚 From: {', '.join(response.sources)}")
            print(f"🏆 Score: {metrics['rag_score']:.2f}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n✅ Quick test complete!")
    print(f"💡 Edit the queries in this file to test your specific questions")


def test_single_query():
    """Test a single custom query"""

    # MODIFY THIS QUERY TO TEST YOUR SPECIFIC QUESTION
    YOUR_QUERY = "What are the key points mentioned in the documents?"

    print(f"🔍 Testing: {YOUR_QUERY}")

    from simple_rag_agent import SimpleRAGAgent
    from vector_store_manager import VectorStoreManager

    vector_manager = VectorStoreManager()
    rag_agent = SimpleRAGAgent(vector_manager)

    try:
        response = rag_agent.query(YOUR_QUERY, k=3)

        print(f"\n📝 Full Response:")
        print(response.response)
        print(f"\n📚 Sources: {', '.join(response.sources)}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Choose what to run:
    quick_simple_test()  # Quick automated test
    # test_single_query()   # Test one specific query