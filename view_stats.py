import pandas as pd
import matplotlib.pyplot as plt

stats_file = "face_expression_stats.csv"

df = pd.read_csv(stats_file)

expressions = ["Natural", "Happy", "Surprised"]
expression_totals = df[expressions].sum()

print("\nOverall Expression Totals:")
print(expression_totals)

custom_labels = [
    f"{expr} ({count} / {expression_totals.sum()})"
    for expr, count in zip(expressions, expression_totals)
]

plt.figure(figsize=(8, 8))
plt.pie(expression_totals, labels=custom_labels, autopct="%1.1f%%", startangle=90)
plt.title("Overall Expression Distribution")
plt.axis("equal")
plt.show()
