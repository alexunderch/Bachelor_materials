#include "node.h"

Node::Node() {};

EmptyNode::EmptyNode() {};
bool EmptyNode::Evaluate(const Date& date, const string& event) {return true;};

DateComparisonNode::DateComparisonNode(const Comparison& cmp, const Date& date) :  _cmp(cmp), _date(date)
{};
bool DateComparisonNode::Evaluate(const Date& date, const string& event)
{
  return Compare(_cmp, date, _date);
}

EventComparisonNode::EventComparisonNode(const Comparison& cmp, const string& event) : _cmp(cmp), _event(event)
{};
bool EventComparisonNode::Evaluate(const Date& date, const string& event)
{
  return Compare(_cmp, event, _event);
}

LogicalOperationNode::LogicalOperationNode(const LogicalOperation& op, const shared_ptr<Node> lhs, const shared_ptr<Node> rhs)
  : _op(op), _lhs(lhs), _rhs(rhs) {};
bool LogicalOperationNode::Evaluate(const Date& date, const string& event)
{
  if (_op  == LogicalOperation::And)
  {
    return _lhs -> Evaluate(date, event) && _rhs -> Evaluate(date, event);
  }
    return _lhs -> Evaluate(date, event) || _rhs -> Evaluate(date, event);
}
