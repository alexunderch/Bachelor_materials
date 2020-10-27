#pragma once
#include "date.h"
#include <string>
#include <memory>
using namespace std;

enum class Comparison
{
  Equal,
  NotEqual,
  Less,
  Greater,
  LessOrEqual,
  GreaterOrEqual,
};

enum class LogicalOperation
{
  And,
  Or,
};

template <typename T>
bool Compare (Comparison cmp, const T& lhs, const T& rhs)
{
  if (cmp == Comparison::Equal)
  {
    return lhs == rhs;
  }
  else if (cmp == Comparison::NotEqual)
  {
    return lhs != rhs;
  }
  else if (cmp == Comparison::Less)
  {
    return lhs < rhs;
  }
  else if (cmp == Comparison::Greater)
  {
    return lhs > rhs;
  }
  else if (cmp == Comparison::LessOrEqual)
  {
    return lhs <= rhs;
  }
  else if (cmp == Comparison::GreaterOrEqual)
  {
    return lhs >= rhs;
  }
  else
  {
    throw logic_error("Unknown Comparison type in node.h");
  }
}

class Node
{
public:
  Node();
  virtual bool Evaluate (const Date& date, const string& event) = 0;
};

class EmptyNode : public Node
{
public:
  EmptyNode();
  bool Evaluate (const Date& date, const string& event) override;
};

class DateComparisonNode : public Node
{
public:
  DateComparisonNode(const Comparison& cmp, const Date& date);
  bool Evaluate (const Date& date, const string& event) override;
private:
  const Comparison _cmp;
  const Date _date;
};

class EventComparisonNode : public Node
{
public:
  EventComparisonNode( const Comparison& cmp, const string& event);
  bool Evaluate (const Date& date, const string& event) override;
private:
  const Comparison _cmp;
  const string _event;
};

class LogicalOperationNode : public Node
{
public:
  LogicalOperationNode(const LogicalOperation& op, const shared_ptr<Node> lhs, const shared_ptr<Node> rhs);
  bool Evaluate (const Date& date, const string& event) override;
private:
  const LogicalOperation _op;
  const shared_ptr<Node> _lhs, _rhs;
};
