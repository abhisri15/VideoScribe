def display_sentiment_text(sentiment):
    with open('sentiment_results.txt', 'r') as file:
        lines = file.readlines()

        in_text = False
        current_sentiment = None

        for line in lines:
            if "Text:" in line:
                in_text = True
                print(line.strip().replace("Text: ", ""))
            elif in_text and "Sentiment:" in line:
                current_sentiment = line.strip().replace("Sentiment: ", "")
            elif in_text and "Timestamp:" in line:
                in_text = False
                if current_sentiment.lower() == sentiment.lower():
                    print(f"Sentiment: {current_sentiment}")
                    print("---")

if __name__ == "__main__":
    user_input = input("Enter sentiment ('ne' for negative, 'p' for positive, 'n' for neutral): ")

    if user_input.lower() in ['ne', 'p', 'n']:
        display_sentiment_text(user_input)
    else:
        print("Invalid input. Please enter 'ne', 'p', or 'n'.")
