import arxiv

def fetch_papers(topic, max_results=3):
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return [result for result in search.results()]

papers = fetch_papers("neural networks")
for p in papers:
    print(p.title, p.summary[:200])
