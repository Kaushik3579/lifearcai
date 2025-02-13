import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

# Load dataset
file_path = "scenario_planning.xlsx"

try:
    df = pd.read_excel(file_path)
    print("✅ File loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found. Please check the path.")
    exit()

# Scenario-based suggestions
event_suggestions = {
    "child_education": "Consider starting an education fund or investing in long-term savings plans.",
    "child_marriage": "Plan early with investments in gold, fixed deposits, or mutual funds.",
    "buying_car": "Check loan options and balance EMIs within 15-20% of your monthly income.",
    "medical_emergency": "Ensure you have sufficient health insurance and an emergency fund.",
    "home_renovation": "Consider cost-effective renovation plans and assess mortgage or personal loan options.",
    "job_loss": "Create an emergency fund covering at least 6 months of expenses and reduce discretionary spending.",
    "vacation_planning": "Save in advance using recurring deposits or travel funds to avoid financial strain.",
    "retirement_planning": "Increase investments in pension plans, long-term funds, and diversify for secure post-retirement life.",
    "starting_a_business": "Assess startup costs, secure funding sources, and manage financial risks wisely.",
    "buying_a_house": "Check mortgage options, calculate EMI affordability, and plan down payments accordingly."
}

# Event-based investment recommendations
event_investments = {
    "child_education": ["mutual_funds", "fixed_deposits", "stocks"],
    "child_marriage": ["gold", "fixed_deposits", "real_estate"],
    "buying_car": ["stocks", "mutual_funds"],
    "medical_emergency": ["insurance", "fixed_deposits"],
    "home_renovation": ["real_estate", "bonds"],
    "job_loss": ["emergency_fund", "fixed_deposits"],
    "vacation_planning": ["recurring_deposits", "mutual_funds"],
    "retirement_planning": ["pension_plans", "mutual_funds", "bonds"],
    "starting_a_business": ["stocks", "real_estate"],
    "buying_a_house": ["real_estate", "fixed_deposits"]
}

investment_options = {
    "stocks": "Invest in stable blue-chip stocks with long-term growth potential.",
    "mutual_funds": "Diversify with mutual funds for balanced risk management.",
    "real_estate": "Invest in real estate for long-term appreciation and rental income.",
    "fixed_deposits": "Secure guaranteed returns with fixed deposits.",
    "gold": "Gold is a safe hedge against inflation and economic downturns.",
    "bonds": "Invest in government and corporate bonds for stable income.",
    "insurance": "Ensure financial security with health and life insurance.",
    "emergency_fund": "Maintain a liquid fund for unexpected expenses.",
    "pension_plans": "Secure your retirement with long-term pension plans.",
    "recurring_deposits": "Save systematically with recurring deposits."
}

# Train regression model for expense prediction
def train_regression_model(user_data):
    expense_history = user_data['Expenses'].values
    X = np.array(range(len(expense_history))).reshape(-1, 1)
    y = np.array(expense_history).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict next month's expenses
def predict_expense(model, next_month):
    return model.predict(np.array([[next_month]])).flatten()[0]

# Suggest loan if user cannot afford the event
def suggest_loan(amount_needed, income):
    interest_rate = 0.08  # 8% annual interest
    time_period = min(5, max(1, int(amount_needed / (0.2 * income))))  # Between 1 to 5 years
    emi = (amount_needed * interest_rate * (1 + interest_rate) ** time_period) / ((1 + interest_rate) ** time_period - 1)
    return round(amount_needed, 2), round(emi, 2), time_period, interest_rate * 100  # Include interest percentage

# Reinforcement Learning Model for Financial Planning
class FinancialQLearning:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.q_table = {}
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
    
    def get_q_values(self, state):
        return self.q_table.get(state, {action: 0 for action in self.actions})
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)
        else:
            q_values = self.get_q_values(state)
            return max(q_values, key=q_values.get)
    
    def update_q_table(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        max_future_q = max(self.get_q_values(next_state).values())
        q_values[action] = q_values[action] + self.learning_rate * (reward + self.discount_factor * max_future_q - q_values[action])
        self.q_table[state] = q_values
        self.exploration_rate *= self.exploration_decay

financial_actions = ["Increase Savings", "Reduce Expenses", "Invest in Stable Assets", "Get Insurance"]
q_model = FinancialQLearning(financial_actions)

def analyze_user_budget(user_id, month):
    user_data = df[df['User_ID'] == user_id]
    if month not in user_data["Month"].values:
        print(f"❌ Warning: No data available for User {user_id}, Month {month}.")
        return None

    income = user_data[user_data['Month'] == month]['Income'].values[0]
    expenses = user_data[user_data['Month'] == month]['Expenses'].values[0]

    # Let the user select an event
    print("\nAvailable Events:")
    for event in event_suggestions:
        print(f"- {event}")
    selected_event = input("Select an event from the list above: ").strip().lower()

    if selected_event not in event_suggestions:
        print("❌ Invalid event selected. Please try again.")
        return None

    event_cost = float(input(f"Enter the estimated cost for '{selected_event}': "))

    model = train_regression_model(user_data)
    predicted_expense = predict_expense(model, month + 1)
    remaining_balance = income - expenses
    can_afford = remaining_balance >= event_cost

    print(f"\nIncome: ${income:,.2f}, Expenses: ${expenses:,.2f}, Event: {selected_event}, Cost: ${event_cost:,.2f}")
    print(event_suggestions.get(selected_event, "No specific recommendation available."))

    # Event-based investment recommendations
    print("\nInvestment Recommendations for the Event:")
    investments = event_investments.get(selected_event, [])
    for inv in investments:
        print(f"{inv.title()}: {investment_options.get(inv, 'No description available.')}")

    # Loan suggestion if the user cannot afford the event
    if not can_afford:
        loan_amount, emi, time_period, interest_rate = suggest_loan(event_cost - remaining_balance, income)
        print(f"\nLoan Details:")
        print(f"Loan Amount: ${loan_amount:,.2f}, EMI: ${emi:,.2f}, Duration: {time_period} years, Interest Rate: {interest_rate}%")

# Function to compare original and changed expenses
def compare_expenses():
    print("\nAvailable Events:")
    for event in event_suggestions:
        print(f"- {event}")
    selected_event = input("Select an event from the list above: ").strip().lower()

    if selected_event not in event_suggestions:
        print("❌ Invalid event selected. Please try again.")
        return None

    original_expense = float(input(f"Enter the original expense for '{selected_event}': "))
    changed_expense = float(input(f"Enter the changed expense for '{selected_event}': "))

    if changed_expense > original_expense:
        change = "increased"
    elif changed_expense < original_expense:
        change = "decreased"
    else:
        change = "remained the same"

    print(f"\nThe expense for '{selected_event}' has {change} from ${original_expense:,.2f} to ${changed_expense:,.2f}.")

    # Provide reasons for the change
    reasons = {
        "child_education": "Tuition fees may have increased due to inflation or additional courses.",
        "child_marriage": "Costs may vary based on venue, number of guests, and other factors.",
        "buying_car": "Car prices can fluctuate due to market demand, new models, or taxes.",
        "medical_emergency": "Medical costs can vary based on treatment type and hospital charges.",
        "home_renovation": "Renovation costs depend on materials, labor, and scope of work.",
        "job_loss": "Expenses may change due to reduced income or increased job search costs.",
        "vacation_planning": "Travel costs can vary based on destination, season, and booking time.",
        "retirement_planning": "Investment returns and inflation can affect retirement savings.",
        "starting_a_business": "Startup costs can vary based on industry, location, and scale.",
        "buying_a_house": "Real estate prices can fluctuate due to market conditions and location."
    }

    print(f"Reason for change: {reasons.get(selected_event, 'No specific reason available.')}")

# Example usage
user_id = input("Enter User ID: ")
month = int(input("Enter Month: "))
analyze_user_budget(user_id, month)

# Compare expenses for a selected event
compare_expenses()
