from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import google.genai as genai
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def list_models():
    """List all available models from Google Generative AI API"""
    print("Available Models:")
    print("-" * 50)
    for model in client.models.list():
        print(f"Model: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print()

def main():
    
    print("Hello from langchain-hello-world!")
    print("Loading environment variables from .env file...")
    # print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

    prompt_template_str = """
    Given the following information about a person {information}, please provide:
        1. A brief summary of the person's background and interests.
        2. Three key points about the information.
    """
    
    information = """Pablo Ruiz Picasso[a][b] (25 October 1881 – 8 April 1973) was a Spanish painter, sculptor, printmaker, ceramicist, and theatre designer who spent most of his adult life in France. One of the most influential artists of the 20th century, he is known for co-founding the Cubist movement, the invention of constructed sculpture,[8][9] the co-invention of collage, and for the wide variety of styles that he helped develop and explore. Among his most famous works are the proto-Cubist Les Demoiselles d'Avignon (1907) and the anti-war painting Guernica (1937), a dramatic portrayal of the bombing of Guernica by German and Italian air forces during the Spanish Civil War. His career spanned more than 76 years, from his late teens to his death in 1973.

        Beginning his formal training under his father José Ruiz y Blasco aged seven, Picasso demonstrated extraordinary artistic talent from a young age, painting in a naturalistic manner through his childhood and adolescence. During the first decade of the 20th century, his style changed as he experimented with different theories, techniques, and ideas. After 1906, the Fauvist work of the older artist Henri Matisse motivated Picasso to explore more radical styles, beginning a fruitful rivalry between the two artists, who subsequently were often paired by critics as the leaders of modern art.[10][11][12][13]

        Picasso's output, especially in his early career, is often periodized. While the names of many of his later periods are debated, the most commonly accepted periods in his work are the Blue Period (1901–1904), the Rose Period (1904–1906), the African-influenced Period (1907–1909), Analytic Cubism (1909–1912), and Synthetic Cubism (1912–1919), also referred to as the Crystal period. Much of Picasso's work of the late 1910s and early 1920s is in a neoclassical style, and his work in the mid-1920s often has characteristics of Surrealism. His later work often combines elements of his earlier styles.

        Exceptionally prolific throughout the course of his long life, Picasso achieved universal renown and immense fortune for his revolutionary artistic accomplishments, and became one of the best-known figures in 20th-century art.
    """
    prompt_template = PromptTemplate(
        input_variables=["information"],
        template=prompt_template_str
    )

    #llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    #llm = ChatOllama(model="gemma3:270m", temperature=0.7)

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.7)

    chain = prompt_template | llm

    response = chain.invoke({"information": information})

    #print("Response from Google Gemini:", response)

    #print("Response from Ollama:", response.content)

    print("Response from Open AI:", response.content)





if __name__ == "__main__":
    # Uncomment to list available models first:
    # list_models()
    main()
