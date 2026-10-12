"""Diagnostic: is the Derrickson paragraph in the index, and do the follow-up queries find it?"""
from src.retriever import Retriever

r = Retriever()

print("Titles containing 'Derrickson' in the index:")
print([d["title"] for d in r.docs if "derrickson" in d["title"].lower()])

for q in ['"Scott Derrickson" nationality', 'Scott Derrickson nationality', "Scott Derrickson"]:
    print(f"\nQUERY: {q}")
    for p in r.search(q, 5):
        print(f"  {p['score']:.2f}  {p['title']}")