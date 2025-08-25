const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json());

/**
 * Simple financial coaching logic:
 * Receives data like income, expenses array and returns suggestions.
 */
app.post('/api/analyze', (req, res) => {
  const { income, expenses } = req.body;
  if (!income || !Array.isArray(expenses)) {
    return res.status(400).json({ error: 'Invalid input data' });
  }

  const totalExpenses = expenses.reduce((a, b) => a + b, 0);
  const savings = income - totalExpenses;
  let advice = [];

  if (savings <= 0) {
    advice.push('Your expenses exceed or match your income. Consider reducing unnecessary spending.');
  } else if (savings < income * 0.1) {
    advice.push('Try to save at least 10% of your income for emergencies.');
  } else {
    advice.push('Great job! You are saving money each month. Consider investing or building an emergency fund.');
  }

  if (expenses.length > 0) {
    const avgExpense = totalExpenses / expenses.length;
    if (avgExpense > income * 0.5) {
      advice.push('Your average expense per transaction is quite high. Track and analyze big expenses carefully.');
    }
  }

  res.json({
    totalExpenses,
    savings,
    advice,
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
