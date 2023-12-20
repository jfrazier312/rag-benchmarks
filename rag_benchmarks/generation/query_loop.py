def query_loop(chain):
    while True:
        query = input("Enter a question:\n")
        response = chain.invoke(query)
        answer = response["answer"]
        print(f"Response:\n{answer}")
        
