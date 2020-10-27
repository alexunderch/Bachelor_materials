#include <iostream>
#include <deque>
#include <string>
#include <vector>
#include <iterator>
using namespace std;

void Expression(const vector<string>& expr)
{
  deque<string> result;
  if (expr.size() > 1)
  {
    for(size_t el = 0; el < expr.size() - 1; el++)
    {
      result.push_front("(");
      result.push_back(expr[el]);
      result.push_back(") ");
    }
    result.push_back(expr[expr.size() - 1]);
    copy(result.begin(), result.end(), ostream_iterator<string>(cout,"") );
  }
}

int main()
{
  int basis = 0;
  cin >> basis;

  int number_of_operations = 0;
  cin >> number_of_operations;

  string op_code;
  if (number_of_operations)
  {
    vector<string> expr_to_build;
    expr_to_build.push_back(to_string(basis));
    getline(cin, op_code);
    while(number_of_operations)
    {
      getline(cin, op_code);
      expr_to_build.push_back(op_code);
      --number_of_operations;
    }
    Expression(expr_to_build);
  }
  else cout << basis;
  cout << endl;
  return 0;
}
