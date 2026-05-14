import numpy as np
from sklearn.ensemble import IsolationForest
import datetime


# ============================================================
#                    DEMO PLATFORM
#       Isolation Forest Fraud Detection + Financial Report
# ============================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_divider():
    print("-" * 60)


# ------------------------------------------------------------
# DEMO 1: Isolation Forest Fraud Detection
# ------------------------------------------------------------

def fraud_detection_demo():
    print_header("ISOLATION FOREST FRAUD DETECTION")
    print("\nThis model detects anomalous transactions that may indicate fraud.")
    print("Enter transaction amounts to build a dataset, then see predictions.\n")

    transactions = []
    transaction_ids = []

    # Collect transactions
    print("Enter transaction amounts (type 'done' when finished):")
    print("Tip: Enter mostly normal values (100-500) and a few outliers (5000+)\n")

    counter = 1
    while True:
        user_input = input(f"  Transaction #{counter} amount: ").strip()

        if user_input.lower() == 'done':
            if len(transactions) < 5:
                print("  ⚠ Need at least 5 transactions. Keep entering.\n")
                continue
            break

        try:
            amount = float(user_input)
            transactions.append(amount)
            transaction_ids.append(f"TXN-{counter:04d}")
            counter += 1
        except ValueError:
            print("  ⚠ Invalid input. Enter a number or 'done'.\n")

    # Train model
    print_divider()
    print("\n🔄 Training Isolation Forest model...")

    X = np.array(transactions).reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)

    # Predict
    predictions = model.predict(X)
    scores = model.decision_function(X)

    # Display results
    print("\n✅ Model trained successfully!\n")
    print_divider()
    print(f"{'ID':<12} {'Amount':>12} {'Status':<12} {'Anomaly Score':>14}")
    print_divider()

    fraud_count = 0
    for i, (tid, amount, pred, score) in enumerate(zip(transaction_ids, transactions, predictions, scores)):
        status = "🚨 FRAUD" if pred == -1 else "✓ Normal"
        if pred == -1:
            fraud_count += 1
        print(f"{tid:<12} ${amount:>10.2f}  {status:<12} {score:>14.4f}")

    print_divider()
    print(f"\n📊 Summary: {fraud_count} suspicious transaction(s) detected out of {len(transactions)}")
    print(f"   Detection rate: {(fraud_count / len(transactions)) * 100:.1f}%")

    # Real-time scoring demo
    print("\n" + "=" * 60)
    print("  REAL-TIME SCORING")
    print("=" * 60)
    print("\nTest new transactions against the trained model.")
    print("Type 'exit' to return to main menu.\n")

    while True:
        user_input = input("  Enter new transaction amount: ").strip()

        if user_input.lower() == 'exit':
            break

        try:
            amount = float(user_input)
            pred = model.predict([[amount]])[0]
            score = model.decision_function([[amount]])[0]

            if pred == -1:
                print(f"  🚨 ALERT: ${amount:.2f} flagged as SUSPICIOUS (score: {score:.4f})\n")
            else:
                print(f"  ✓ ${amount:.2f} appears normal (score: {score:.4f})\n")
        except ValueError:
            print("  ⚠ Invalid input.\n")


# ------------------------------------------------------------
# DEMO 2: Financial Report Generator
# ------------------------------------------------------------

def financial_report_demo():
    print_header("AI FINANCIAL REPORT GENERATOR")
    print("\nGenerate professional financial summaries from your data.\n")

    # Collect inputs
    print("Enter financial data:\n")

    while True:
        try:
            revenue = float(input("  Total Revenue ($): "))
            break
        except ValueError:
            print("  ⚠ Enter a valid number.\n")

    while True:
        try:
            expenses = float(input("  Total Expenses ($): "))
            break
        except ValueError:
            print("  ⚠ Enter a valid number.\n")

    profit = revenue - expenses
    profit_margin = (profit / revenue * 100) if revenue > 0 else 0

    # Generate report
    print_divider()
    print("\n🔄 Generating AI-powered financial report...\n")

    # Simulated LLM prompt construction (shown for demo)
    prompt = f"""
    Generate a financial summary:
    Revenue: ${revenue:,.2f}
    Expenses: ${expenses:,.2f}
    Profit: ${profit:,.2f}
    """

    print("📝 LLM Prompt Sent:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)

    # Simulated LLM response
    if profit > 0:
        performance = "profitable"
        recommendation = "Consider reinvesting surplus into growth initiatives."
    elif profit < 0:
        performance = "operating at a loss"
        recommendation = "Immediate cost optimization review recommended."
    else:
        performance = "breaking even"
        recommendation = "Focus on revenue growth strategies."

    report = f"""
╔══════════════════════════════════════════════════════════╗
║              FINANCIAL SUMMARY REPORT                    ║
║              Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  REVENUE:        ${revenue:>15,.2f}                      
║  EXPENSES:       ${expenses:>15,.2f}                     
║  ─────────────────────────────────                       
║  NET PROFIT:     ${profit:>15,.2f}                       
║  PROFIT MARGIN:  {profit_margin:>14.1f}%                        
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  ANALYSIS                                                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  The organization is currently {performance}.            
║                                                          ║
║  Revenue of ${revenue:,.2f} against expenses of          
║  ${expenses:,.2f} yields a net {"surplus" if profit >= 0 else "deficit"} of ${abs(profit):,.2f}.
║                                                          ║
║  RECOMMENDATION: {recommendation}                        
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""

    print("\n🤖 LLM Response:")
    print(report)

    # Option to generate another
    again = input("\nGenerate another report? (y/n): ").strip().lower()
    if again == 'y':
        financial_report_demo()


# ------------------------------------------------------------
# MAIN MENU
# ------------------------------------------------------------

def main():
    while True:
        print_header("AI/ML DEMO PLATFORM")
        print("""
  Select a demo to run:

  [1] Isolation Forest Fraud Detection
      → Train model on manual transaction inputs
      → Real-time anomaly scoring

  [2] AI Financial Report Generator
      → Input revenue/expenses manually
      → Generate formatted financial summary

  [3] Exit
""")

        choice = input("  Enter choice (1/2/3): ").strip()

        if choice == '1':
            fraud_detection_demo()
        elif choice == '2':
            financial_report_demo()
        elif choice == '3':
            print("\n  Goodbye!\n")
            break
        else:
            print("\n  ⚠ Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()
