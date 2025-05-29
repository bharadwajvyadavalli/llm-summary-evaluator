"""
Simple Test System - Test any PDF documents with any queries
"""

import os
import pandas as pd
from simple_pdf_processor import SimplePDFProcessor
from vector_store_manager import VectorStoreManager
from simple_rag_agent import SimpleRAGAgent
from rag_evaluator import RAGEvaluator


def get_sample_test_queries():
    """Sample queries you can modify based on your documents"""

    return [
        # General content queries
        "What is this document about?",
        "What are the main topics covered?",
        "What are the key points mentioned?",

        # Specific information queries
        "What details are provided about the main subject?",
        "What examples or cases are discussed?",
        "What recommendations or suggestions are made?",

        # Analysis queries
        "What problems or challenges are identified?",
        "What solutions or approaches are described?",
        "What conclusions are reached?",

        # Custom queries (modify these based on your documents)
        "What specific information is most important?",
        "What facts or data are presented?"
    ]


def simple_rag_test():
    """Simple RAG test with any documents and queries"""

    print("🚀 SIMPLE RAG TEST")
    print("=" * 30)

    # Step 1: Initialize simple components
    print("\n1️⃣ Initializing RAG system...")

    try:
        processor = SimplePDFProcessor()
        vector_manager = VectorStoreManager()
        rag_agent = SimpleRAGAgent(vector_manager)
        evaluator = RAGEvaluator()

        print("✅ Simple RAG system initialized")

    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return

    # Step 2: Process PDF documents
    print("\n2️⃣ Processing PDF documents...")

    stats = vector_manager.get_collection_stats()

    if stats['total_chunks'] == 0:
        pdf_directory = "sample_pdfs"

        if not os.path.exists(pdf_directory):
            print(f"❌ Directory '{pdf_directory}' not found")
            print("💡 Create the directory and add any PDF files")
            return

        try:
            documents = processor.process_pdfs(pdf_directory)

            if documents:
                vector_manager.add_documents(documents)
                print(f"✅ Processed {len(documents)} documents")

                # Show what was processed
                print("\n📄 Document Summary:")
                for doc in documents:
                    print(f"   • {doc['document_name']}: {doc['word_count']:,} words")

            else:
                print("❌ No documents were processed")
                return

        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            return
    else:
        print(f"✅ Using existing {stats['total_chunks']} chunks")

    # Step 3: Test with sample queries
    print("\n3️⃣ Testing with sample queries...")

    test_queries = get_sample_test_queries()

    # You can modify this to test specific queries
    selected_queries = test_queries[:5]  # Test first 5 queries

    print(f"🧪 Running {len(selected_queries)} test queries...")

    results = []

    for i, query in enumerate(selected_queries, 1):
        print(f"\n--- Query {i}/{len(selected_queries)} ---")
        print(f"❓ {query}")

        try:
            # Process query
            response = rag_agent.query(query, k=3)

            # Evaluate response
            metrics = evaluator.evaluate_response(response)

            # Store results
            result = {
                'query': query,
                'response': response.response,
                'sources': ', '.join(response.sources),
                'num_chunks': len(response.context_chunks),
                'response_length': len(response.response),
                **metrics
            }
            results.append(result)

            # Show results
            print(f"📝 Response: {response.response[:150]}...")
            print(f"📚 Sources: {', '.join(response.sources)}")
            print(f"🏆 RAG Score: {metrics['rag_score']:.3f}")

            # Simple quality indicator
            if metrics['rag_score'] >= 0.7:
                quality = "🟢 GOOD"
            elif metrics['rag_score'] >= 0.5:
                quality = "🟡 OK"
            else:
                quality = "🔴 NEEDS IMPROVEMENT"

            print(f"✅ Quality: {quality}")

        except Exception as e:
            print(f"❌ Query failed: {e}")
            continue

    # Step 4: Summary and results
    print("\n4️⃣ Test Results Summary")
    print("=" * 30)

    if results:
        df = pd.DataFrame(results)

        # Save results
        df.to_csv("simple_rag_results.csv", index=False)
        print("💾 Results saved to: simple_rag_results.csv")

        # Summary statistics
        print(f"\n📊 Performance Summary:")
        print(f"   • Queries tested: {len(results)}")
        print(f"   • Average RAG Score: {df['rag_score'].mean():.3f}")
        print(f"   • Average Response Length: {df['response_length'].mean():.0f} chars")
        print(f"   • Documents used: {len(set(df['sources'].sum().split(', ')))}")

        # Best performing query
        best_idx = df['rag_score'].idxmax()
        print(f"\n🏆 Best Query (Score: {df.loc[best_idx, 'rag_score']:.3f}):")
        print(f"   '{df.loc[best_idx, 'query']}'")

        # Quality distribution
        good = len(df[df['rag_score'] >= 0.7])
        ok = len(df[(df['rag_score'] >= 0.5) & (df['rag_score'] < 0.7)])
        poor = len(df[df['rag_score'] < 0.5])

        print(f"\n📈 Quality Distribution:")
        print(f"   • Good (≥0.7): {good} queries")
        print(f"   • OK (0.5-0.7): {ok} queries")
        print(f"   • Poor (<0.5): {poor} queries")

    else:
        print("❌ No results to analyze")

    print(f"\n✅ Simple RAG test complete!")


def custom_query_test():
    """Test with your own custom queries"""

    print("\n🎯 CUSTOM QUERY TEST")
    print("=" * 25)

    # Initialize system
    vector_manager = VectorStoreManager()
    rag_agent = SimpleRAGAgent(vector_manager)

    # Add your custom queries here
    custom_queries = [
        "YOUR CUSTOM QUERY 1",
        "YOUR CUSTOM QUERY 2",
        "YOUR CUSTOM QUERY 3"
    ]

    print("💡 Modify the custom_queries list in the code to test your specific questions")

    for i, query in enumerate(custom_queries, 1):
        print(f"\n🔍 Query {i}: {query}")

        try:
            response = rag_agent.query(query, k=3)
            print(f"📝 Response: {response.response[:200]}...")
            print(f"📚 Sources: {', '.join(response.sources)}")

        except Exception as e:
            print(f"❌ Error: {e}")


def interactive_query_test():
    """Interactive testing - type your own queries"""

    print("\n💬 INTERACTIVE QUERY TEST")
    print("=" * 30)

    # Initialize system
    vector_manager = VectorStoreManager()
    rag_agent = SimpleRAGAgent(vector_manager)
    evaluator = RAGEvaluator()

    print("Type your queries (press Enter twice to finish):")

    while True:
        query = input("\n❓ Your query: ").strip()

        if not query:
            break

        try:
            response = rag_agent.query(query, k=3)
            metrics = evaluator.evaluate_response(response)

            print(f"\n📝 Response: {response.response}")
            print(f"📚 Sources: {', '.join(response.sources)}")
            print(f"🏆 Score: {metrics['rag_score']:.3f}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("✅ Interactive testing finished!")


if __name__ == "__main__":
    # Choose which test to run:
    simple_rag_test()  # Automated test with sample queries
    # custom_query_test()    # Test with your custom queries
    # interactive_query_test() # Type queries interactively