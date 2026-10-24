from routing import route_ticket


def main():
    examples = [
        "My bank transfer has not arrived.",
        "I am still waiting for my bank card.",
        "I need help with something in my account.",
    ]

    for text in examples:
        result = route_ticket(text)

        print("\n" + "=" * 60)
        print("Ticket:", result["ticket_text"])
        print("Predicted intent:", result["predicted_intent"])
        print(f"Confidence: {result['confidence']:.4f}")
        print("Proposed department:", result["proposed_department"])
        print("Decision:", result["decision"])
        print("Assigned department:", result["assigned_department"])
        print("Reason:", result["reason"])


if __name__ == "__main__":
    main()