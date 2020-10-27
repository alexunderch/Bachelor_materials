#include "Common.h"
#include "test_runner.h"
#include <functional>
#include <sstream>

using namespace std;

class ValueExpresion : public Expression {
private:
  int val_;
  public:
  ValueExpresion (int val) : val_(val) {}
  int Evaluate() const override { return val_; }
  string ToString() const override {
    return to_string(val_);
  }
};

class BinaryExpression : public Expression {
  private:
  virtual char GetSymbol() const = 0;
  virtual int EvaluateOnValues(int lhs, int rhs) const = 0;
  ExpressionPtr left_;
  ExpressionPtr right_;
public:
  BinaryExpression (ExpressionPtr left, ExpressionPtr right) : left_(move(left)),
                                                                right_(move(right)){}
  string ToString() const final {
    return '(' + left_ -> ToString() + ')' + GetSymbol() +  
           '(' + right_ -> ToString() + ')';
  }
  int Evaluate () const final {
    return EvaluateOnValues(left_ -> Evaluate(), right_ -> Evaluate());
  }
};

class ProductExpr : public BinaryExpression {
public:
  ProductExpr(ExpressionPtr left, ExpressionPtr right) : BinaryExpression(move(left), move(right)) {}
  
private:
  char GetSymbol() const override { return '*'; }
  int EvaluateOnValues(int lhs, int rhs) const { return lhs * rhs; }

};

class SumExpr : public BinaryExpression {
public:
  using BinaryExpression::BinaryExpression;
 private:
  char GetSymbol() const override { return '+'; }
  int EvaluateOnValues(int lhs, int rhs) const { return lhs + rhs; }
};

ExpressionPtr Value(int value) {
  return make_unique<ValueExpresion>(value); 
}
ExpressionPtr Sum(ExpressionPtr left, ExpressionPtr right) {
  return make_unique<SumExpr> (move(left), move(right));
}
ExpressionPtr Product(ExpressionPtr left, ExpressionPtr right) {
  return make_unique<ProductExpr> (move(left), move(right));
}



string Print(const Expression* e) {
  if (!e) {
    return "Null expression provided";
  }
  stringstream output;
  output << e->ToString() << " = " << e->Evaluate();
  return output.str();
}

void Test() {
  ExpressionPtr e1 = Product(Value(2), Sum(Value(3), Value(4)));
  ASSERT_EQUAL(Print(e1.get()), "(2)*((3)+(4)) = 14");

  ExpressionPtr e2 = Sum(move(e1), Value(5));
  ASSERT_EQUAL(Print(e2.get()), "((2)*((3)+(4)))+(5) = 19");

  ASSERT_EQUAL(Print(e1.get()), "Null expression provided");
}

int main() {
  TestRunner tr;
  RUN_TEST(tr, Test);
  return 0;
}