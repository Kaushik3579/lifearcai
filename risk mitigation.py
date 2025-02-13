import numpy as np
import random

def get_user_input():
    income = float(input("Enter your monthly income: "))
    primary_expenses = float(input("Enter your primary expenses (rent, utilities, groceries, etc.): "))
    entertainment = float(input("Enter your entertainment expenses: "))
    travel = float(input("Enter your travel expenses: "))
    lifestyle = float(input("Enter your lifestyle expenses (shopping, dining, etc.): "))
    medical = float(input("Enter your medical expenses: "))
    other_expenses = float(input("Enter any other expenses: "))
    inflation_rate = float(input("Enter current inflation rate (%): "))
    event = input("Enter any major financial event (child_education, child_marriage, buying_car, medical_emergency, home_renovation, job_loss, vacation_planning, retirement_planning, starting_a_business, buying_a_house): ")
    
    total_expenses = primary_expenses + entertainment + travel + lifestyle + medical + other_expenses
    
    return {
        "income": income,
        "primary_expenses": primary_expenses,
        "entertainment": entertainment,
        "travel": travel,
        "lifestyle": lifestyle,
        "medical": medical,
        "other_expenses": other_expenses,
        "total_expenses": total_expenses,
        "inflation_rate": inflation_rate,
        "event": event
    }

def analyze_expenses(data):
    suggestions = []
    
    if data['total_expenses'] > data['income']:
        suggestions.append("⚠️ Your total expenses exceed your income. Consider reducing discretionary spending.")
    
    if data['entertainment'] > 0.1 * data['income']:
        suggestions.append("🎭 Reduce entertainment expenses. Consider free or low-cost activities.")
    
    if data['travel'] > 0.1 * data['income']:
        suggestions.append("✈️ Reduce travel costs by opting for budget-friendly alternatives.")
    
    if data['lifestyle'] > 0.15 * data['income']:
        suggestions.append("🛍️ Reduce lifestyle expenses. Avoid impulse shopping and dining out frequently.")
    
    if data['medical'] == 0:
        suggestions.append("🏥 Consider health insurance to prepare for medical emergencies.")
    
    if data['inflation_rate'] > 5:
        suggestions.append("📈 Inflation is high. Invest in assets that grow over time to protect your savings.")
    
    # AI-based financial risk identification
    if data['income'] - data['total_expenses'] < 0.2 * data['income']:
        suggestions.append("🚨 Your savings buffer is low. Consider increasing your emergency fund.")
    
    if data['inflation_rate'] > 6:
        suggestions.append("📉 High inflation risk detected. Diversify investments into inflation-protected assets.")
    
    return suggestions

# Q-Learning for Financial Stability
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

# Event-based suggestions
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

# Initialize Q-learning model with possible financial actions
actions = ["Increase Savings", "Reduce Expenses", "Invest in Stable Assets", "Get Insurance", "Build Emergency Fund"]
q_model = FinancialQLearning(actions)

def get_financial_risks(data):
    state = f"Income:{data['income']} Expenses:{data['total_expenses']} Inflation:{data['inflation_rate']}"
    action = q_model.choose_action(state)
    return action

def main():
    user_data = get_user_input()
    expense_suggestions = analyze_expenses(user_data)
    financial_risk_action = get_financial_risks(user_data)
    event_suggestion = event_suggestions.get(user_data['event'], "No specific recommendation for this event.")
    
    print("\n--- Financial Risk Analysis ---")
    for suggestion in expense_suggestions:
        print(suggestion)
    
    print(f"\n🚀 AI Suggestion for Financial Stability: {financial_risk_action}")
    print(f"\n📌 Event-Based Financial Advice: {event_suggestion}")

if __name__ == "__main__":
    main()
