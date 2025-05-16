import os
from io import StringIO
import pandas as pd
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain_core.prompts import PromptTemplate
import networkx as nx
import matplotlib.pyplot as plt
from langchain_graph_retriever.document_graph import create_graph

# Load environment variables
load_dotenv()

# Sample data as strings (you can replace this with actual file loading)
reviews_data_string = """
id,reviewId,creationDate,criticName,isTopCritic,originalScore,reviewState,publicatioName,reviewText,scoreSentiment,reviewUrl
addams_family,2644238,2019-11-10,James Kendrick,False,3/4,fresh,Q Network Film Desk,captures the family's droll humor with just the right mixture of morbidity and genuine care,POSITIVE,http://www.qnetwork.com/review/4178
addams_family,2509777,2018-09-12,John Ferguson,False,4/5,fresh,Radio Times,A witty family comedy that has enough sly humour to keep adults chuckling throughout.,POSITIVE,https://www.radiotimes.com/film/fj8hmt/the-addams-family/
addams_family,26216,2000-01-01,Rita Kempley,True,,fresh,Washington Post,"More than merely a sequel of the TV series, the film is a compendium of paterfamilias Charles Addams's macabre drawings, a resurrection of the cartoonist's body of work. For family friends, it would seem a viewing is de rigueur mortis.",POSITIVE,http://www.washingtonpost.com/wp-srv/style/longterm/movies/videos/theaddamsfamilypg13kempley_a0a280.htm
"""

movies_data_string = """
id,title,audienceScore,tomatoMeter,rating,ratingContents,releaseDateTheaters,releaseDateStreaming,runtimeMinutes,genre,originalLanguage,director,writer,boxOffice,distributor,soundMix
addams_family,The Addams Family,66,67,,,1991-11-22,2005-08-18,99,Comedy,English,Barry Sonnenfeld,"Charles Addams,Caroline Thompson,Larry Wilson",$111.3M,Paramount Pictures,"Surround, Dolby SR"
the_addams_family_2019,The Addams Family,69,45,PG,"['Some Action', 'Macabre and Suggestive Humor']",2019-10-11,2019-10-11,87,"Kids & family, Comedy, Animation",English,"Conrad Vernon,Greg Tiernan","Matt Lieberman,Erica Rivinoja",$673.0K,Metro-Goldwyn-Mayer,Dolby Atmos
"""

def load_sample_data():
    """Load sample data from strings"""
    reviews_all = pd.read_csv(StringIO(reviews_data_string))
    movies_all = pd.read_csv(StringIO(movies_data_string))
    
    # Rename columns for graph structure
    reviews_data = reviews_all.rename(columns={"id": "reviewed_movie_id"})
    movies_data = movies_all.rename(columns={"id": "movie_id"})
    
    return reviews_data, movies_data

def create_documents(reviews_data, movies_data):
    """Convert data to LangChain documents"""
    documents = []
    
    # Convert movies to documents
    for index, row in movies_data.iterrows():
        content = str(row["title"])
        metadata = row.fillna("").astype(str).to_dict()
        metadata["doc_type"] = "movie_info"
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    # Convert reviews to documents
    for index, row in reviews_data.iterrows():
        content = str(row["reviewText"])
        metadata = row.drop("reviewText").fillna("").astype(str).to_dict()
        metadata["doc_type"] = "movie_review"
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)
    
    return documents

def setup_retriever(vectorstore):
    """Set up the GraphRetriever"""
    return GraphRetriever(
        store=vectorstore,
        edges=[("reviewed_movie_id", "movie_id")],
        strategy=Eager(start_k=10, adjacent_k=10, select_k=100, max_depth=1),
    )

def compile_results(query_results):
    """Compile and format the query results"""
    compiled_results = {}
    
    # Collect movie info
    for result in query_results:
        if result.metadata["doc_type"] == "movie_info":
            movie_id = result.metadata["movie_id"]
            movie_title = result.metadata["title"]
            compiled_results[movie_id] = {
                "movie_id": movie_id,
                "movie_title": movie_title,
                "reviews": {},
            }
    
    # Collect reviews
    for result in query_results:
        if result.metadata["doc_type"] == "movie_review":
            reviewed_movie_id = result.metadata["reviewed_movie_id"]
            review_id = result.metadata["reviewId"]
            review_text = result.page_content
            if reviewed_movie_id in compiled_results:
                compiled_results[reviewed_movie_id]["reviews"][review_id] = review_text
    
    # Format results
    formatted_text = ""
    for movie_id, review_list in compiled_results.items():
        formatted_text += "\n\n Movie Title: "
        formatted_text += review_list["movie_title"]
        formatted_text += "\n Movie ID: "
        formatted_text += review_list["movie_id"]
        for review_id, review_text in review_list["reviews"].items():
            formatted_text += "\n Review: "
            formatted_text += review_text
    
    return formatted_text

def save_graph_visualization(query_results, retriever, output_path="graph_visualization.png"):
    """Create and save graph visualization"""
    document_graph = create_graph(
        documents=query_results,
        edges=retriever.edges,
    )
    
    plt.figure(figsize=(12, 8))
    nx.draw(document_graph, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=8, font_weight='bold')
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGraph visualization saved to: {output_path}")

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Load data
    reviews_data, movies_data = load_sample_data()
    
    # Create vector store
    vectorstore = InMemoryVectorStore(OpenAIEmbeddings())
    
    # Create and add documents
    documents = create_documents(reviews_data, movies_data)
    vectorstore.add_documents(documents)
    
    # Set up retriever
    retriever = setup_retriever(vectorstore)
    
    # Example query
    INITIAL_PROMPT_TEXT = "What are some good family movies?"
    query_results = retriever.invoke(INITIAL_PROMPT_TEXT)
    
    # Save graph visualization
    save_graph_visualization(query_results, retriever)
    
    # Compile results
    formatted_text = compile_results(query_results)
    
    # Set up LLM and prompt
    model = ChatOpenAI(model="gpt-4", temperature=0)
    prompt_template = PromptTemplate.from_template("""
    A list of Movie Reviews appears below. Please answer the Initial Prompt text
    (below) using only the listed Movie Reviews.

    Please include all movies that might be helpful to someone looking for movie
    recommendations.

    Initial Prompt:
    {initial_prompt}

    Movie Reviews:
    {movie_reviews}
    """)
    
    # Generate response
    formatted_prompt = prompt_template.format(
        initial_prompt=INITIAL_PROMPT_TEXT,
        movie_reviews=formatted_text,
    )
    result = model.invoke(formatted_prompt)
    
    print("\nQuery Results:")
    print("-------------")
    print(result.content)

if __name__ == "__main__":
    main() 